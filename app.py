
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime


# ----------------------------
# App settings
# ----------------------------
st.set_page_config(
    page_title="FBS Power Ratings",
    page_icon="🏈",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "cfbd_model_data_2015_2025.csv"
CFBD_BASE_URL = "https://api.collegefootballdata.com"

START_RATING = 1500.0
HOME_ADVANTAGE_ELO = 55.0
K_FACTOR = 20.0


# ----------------------------
# Load historical games
# ----------------------------
@st.cache_data
def load_games(file_modified_time):
    del file_modified_time  # Used to invalidate the cache when the file changes.

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. "
            "Upload the CSV into the same GitHub repository as app.py."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "season",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The CSV is missing these columns: " + ", ".join(sorted(missing))
        )

    df = df.dropna(
        subset=["season", "home_team", "away_team", "home_points", "away_points"]
    ).copy()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df = df.dropna(
        subset=["season", "home_points", "away_points"]
    )

    df["season"] = df["season"].astype(int)
    df["home_points"] = df["home_points"].astype(float)
    df["away_points"] = df["away_points"].astype(float)

    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df = df.sort_values(["start_date", "season"])
    elif "week" in df.columns:
        df = df.sort_values(["season", "week"])
    else:
        df = df.sort_values("season")

    return df.reset_index(drop=True)


# ----------------------------
# Try to obtain the FBS team list
# ----------------------------
@st.cache_data(ttl=3600)
def get_fbs_teams():
    try:
        api_key = st.secrets.get("CFBD_API_KEY", "")
        if not api_key:
            return None

        response = requests.get(
            f"{CFBD_BASE_URL}/teams/fbs",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        names = {
            item.get("school")
            for item in payload
            if isinstance(item, dict) and item.get("school")
        }

        return names or None
    except Exception:
        return None


# ----------------------------
# Elo rating calculations
# ----------------------------
@st.cache_data
def calculate_ratings(games):
    ratings = {}
    records = {}
    rating_rows = []

    seasons = sorted(games["season"].unique())

    previous_season = None

    for _, game in games.iterrows():
        season = int(game["season"])
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        home_score = float(game["home_points"])
        away_score = float(game["away_points"])

        if not home or not away or home == away:
            continue

        # At the beginning of a new season, regress ratings toward average.
        if previous_season is not None and season != previous_season:
            for team in list(ratings):
                ratings[team] = 1500 + 0.75 * (ratings[team] - 1500)

        previous_season = season

        ratings.setdefault(home, START_RATING)
        ratings.setdefault(away, START_RATING)

        records.setdefault(home, {"wins": 0, "losses": 0})
        records.setdefault(away, {"wins": 0, "losses": 0})

        neutral = bool(game.get("neutral_site", False))
        home_edge = 0 if neutral else HOME_ADVANTAGE_ELO

        home_before = ratings[home]
        away_before = ratings[away]

        expected_home = 1 / (
            1 + 10 ** (-((home_before + home_edge - away_before) / 400))
        )

        if home_score > away_score:
            actual_home = 1.0
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif home_score < away_score:
            actual_home = 0.0
            records[home]["losses"] += 1
            records[away]["wins"] += 1
        else:
            actual_home = 0.5

        margin = abs(home_score - away_score)
        margin_factor = math.log1p(margin) * (
            2.2 / (2.2 + 0.001 * abs(home_before - away_before))
        )
        change = K_FACTOR * margin_factor * (actual_home - expected_home)

        ratings[home] += change
        ratings[away] -= change

        rating_rows.append(
            {
                "season": season,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    rankings = pd.DataFrame(
        [{"Team": team, "Rating": rating} for team, rating in ratings.items()]
    )

    if rankings.empty:
        return rankings, pd.DataFrame(), pd.DataFrame()

    rankings["Wins"] = rankings["Team"].map(
        lambda t: records.get(t, {}).get("wins", 0)
    )
    rankings["Losses"] = rankings["Team"].map(
        lambda t: records.get(t, {}).get("losses", 0)
    )

    # Display records for the most recent season present in the file.
    latest_season = int(games["season"].max())
    latest = games[games["season"] == latest_season]
    latest_records = {}

    for _, game in latest.iterrows():
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        hs = float(game["home_points"])
        aws = float(game["away_points"])

        latest_records.setdefault(home, {"W": 0, "L": 0})
        latest_records.setdefault(away, {"W": 0, "L": 0})

        if hs > aws:
            latest_records[home]["W"] += 1
            latest_records[away]["L"] += 1
        elif aws > hs:
            latest_records[away]["W"] += 1
            latest_records[home]["L"] += 1

    rankings["Latest Season W-L"] = rankings["Team"].map(
        lambda t: (
            f'{latest_records.get(t, {}).get("W", 0)}-'
            f'{latest_records.get(t, {}).get("L", 0)}'
        )
    )

    rankings = rankings.sort_values("Rating", ascending=False).reset_index(drop=True)
    rankings.insert(0, "Rank", np.arange(1, len(rankings) + 1))

    return rankings, pd.DataFrame(rating_rows), games


def predict_matchup(team_a, team_b, ratings, games):
    rating_map = ratings.set_index("Team")["Rating"].to_dict()

    elo_a = float(rating_map[team_a])
    elo_b = float(rating_map[team_b])

    # Neutral-site expected margin: positive means Team A is favored.
    margin_a = (elo_a - elo_b) / 25.0

    # Estimate typical scoring from the historical dataset.
    total_points = float((games["home_points"] + games["away_points"]).mean())
    total_points = max(total_points, abs(margin_a) + 14)

    score_a = max(0, (total_points + margin_a) / 2)
    score_b = max(0, (total_points - margin_a) / 2)

    # Elo-based win probability, with a college-football-friendly scale.
    prob_a = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))
    prob_b = 1 - prob_a

    return score_a, score_b, margin_a, prob_a, prob_b


# ----------------------------
# Main dashboard
# ----------------------------
st.title("🏈 FBS Power Ratings")
st.caption(
    "Historical Elo-style ratings and neutral-site matchup estimates. "
    "This is an initial baseline model, not an official ranking."
)

if st.button("🔄 Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

try:
    modified_time = DATA_FILE.stat().st_mtime if DATA_FILE.exists() else 0
    games = load_games(modified_time)
except Exception as error:
    st.error(str(error))
    st.info(
        "Make sure cfbd_model_data_2015_2025.csv is in the repository "
        "beside app.py."
    )
    st.stop()


st.subheader("Live College Football Data")

if st.button("Fetch current season games"):
    try:
        with st.spinner("Fetching games from CollegeFootballData..."):
            live_completed, live_upcoming = fetch_current_season_games()

            st.session_state["live_completed"] = live_completed
            st.session_state["live_upcoming"] = live_upcoming
            st.session_state["live_fetch_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    except Exception as error:
        st.error(f"Could not fetch live games: {error}")

if "live_completed" in st.session_state:
    live_completed = st.session_state["live_completed"]
    live_upcoming = st.session_state["live_upcoming"]

    # Add current-season final scores to the historical dataset.
    games = pd.concat(
        [games, live_completed],
        ignore_index=True,
        sort=False,
    )

    # Remove duplicate games if the refresh is repeated.
    if "id" in games.columns:
        games = games.drop_duplicates(subset=["id"], keep="last")

    st.success(
        f"Live data loaded: {len(live_completed)} games with scores "
        f"and {len(live_upcoming)} scheduled games."
    )

    st.caption(
        "Last fetched: " + st.session_state["live_fetch_time"]
    )

    if not live_upcoming.empty:
        st.subheader("Upcoming Games")
        display_columns = [
            column
            for column in [
                "start_date", "week", "away_team", "home_team"
            ]
            if column in live_upcoming.columns
        ]

        st.dataframe(
            live_upcoming[display_columns].sort_values(
                "start_date"
            ) if "start_date" in display_columns
            else live_upcoming[display_columns],
            use_container_width=True,
            hide_index=True,
        )

rankings, _, _ = calculate_ratings(games)


if rankings.empty:
    st.error("No valid games were found in the CSV.")
    st.stop()

fbs_teams = get_fbs_teams()

if fbs_teams:
    rankings = rankings[rankings["Team"].isin(fbs_teams)].copy()
    st.caption("FBS filter: using the CollegeFootballData team list.")
else:
    st.warning(
        "FBS-only filtering is not active. Add your CFBD_API_KEY in the app's "
        "secrets to filter the table to FBS teams. Without it, the table may "
        "include FCS opponents."
    )


def fetch_current_season_games():
    api_key = st.secrets.get("CFBD_API_KEY", "").strip()

    if not api_key:
        raise ValueError("CFBD_API_KEY is missing from Streamlit Secrets.")

    season = datetime.now().year

    response = requests.get(
        f"{CFBD_BASE_URL}/games",
        params={
            "year": season,
            "seasonType": "both",
            "classification": "fbs",
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            f"CollegeFootballData returned HTTP {response.status_code}. "
            "Check your API access and usage limits."
        )

    payload = response.json()

    if not payload:
        raise ValueError(f"No games were returned for {season}.")

    live = pd.DataFrame(payload)

    # Convert API field names to the names used by the existing model.
    live = live.rename(
        columns={
            "homeTeam": "home_team",
            "awayTeam": "away_team",
            "homePoints": "home_points",
            "awayPoints": "away_points",
            "neutralSite": "neutral_site",
            "startDate": "start_date",
            "seasonType": "season_type",
        }
    )

    required = [
        "home_team",
        "away_team",
        "home_points",
        "away_points",
    ]

    missing = [column for column in required if column not in live.columns]
    if missing:
        raise ValueError(
            "Unexpected API response; missing fields: " + ", ".join(missing)
        )

    live["home_points"] = pd.to_numeric(live["home_points"], errors="coerce")
    live["away_points"] = pd.to_numeric(live["away_points"], errors="coerce")

    # Only final-score games are used to update ratings.
    completed = live.dropna(
        subset=["home_points", "away_points"]
    ).copy()

    # The remaining games are scheduled games without final scores.
    upcoming = live[
        live["home_points"].isna() | live["away_points"].isna()
    ].copy()

    return completed, upcoming


rankings = rankings.sort_values("Rating", ascending=False).reset_index(drop=True)
rankings["Rank"] = np.arange(1, len(rankings) + 1)

latest_season = int(games["season"].max())

col1, col2, col3 = st.columns(3)
col1.metric("Teams in rankings", len(rankings))
col2.metric("Historical games", f"{len(games):,}")
col3.metric("Latest season in file", latest_season)

st.subheader("National Power Rankings")

search = st.text_input("Find a team")
if search:
    shown = rankings[
        rankings["Team"].str.contains(search, case=False, na=False)
    ]
else:
    shown = rankings

st.dataframe(
    shown[["Rank", "Team", "Rating", "Latest Season W-L"]].round(
        {"Rating": 1}
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download rankings as CSV",
    data=rankings.to_csv(index=False).encode("utf-8"),
    file_name="fbs_power_rankings.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Neutral-Site Matchup Predictor")

team_names = sorted(rankings["Team"].tolist())

if len(team_names) >= 2:
    left, right = st.columns(2)

    team_a = left.selectbox("Team A", team_names, index=0)
    team_b = right.selectbox(
        "Team B",
        team_names,
        index=1 if len(team_names) > 1 else 0,
    )

    if team_a == team_b:
        st.info("Choose two different teams to run a prediction.")
    else:
        score_a, score_b, margin, prob_a, prob_b = predict_matchup(
            team_a, team_b, rankings, games
        )

        favored = team_a if margin >= 0 else team_b
        spread = abs(margin)

        st.markdown(f"### Projected score: {team_a} {score_a:.1f} — {team_b} {score_b:.1f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Projected favorite", favored)
        c2.metric("Projected spread", f"{favored} -{spread:.1f}")
        c3.metric("Projected total", f"{score_a + score_b:.1f}")

        p1, p2 = st.columns(2)
        p1.metric(f"{team_a} win probability", f"{prob_a:.1%}")
        p2.metric(f"{team_b} win probability", f"{prob_b:.1%}")

        st.caption(
            "Scores and probabilities are preliminary estimates. "
            "The initial score model uses average historical scoring and "
            "an Elo rating difference; it has not yet been calibrated against "
            "a held-out test season."
        )
else:
    st.warning("At least two teams are needed to make a matchup prediction.")

st.divider()
st.caption(
    "Data file currently covers 2015–2025. Current-season results, live schedules, "
    "weekly rating movement, and model validation will be added in later steps."
)
