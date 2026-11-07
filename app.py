
import math
import html
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
    api_key = st.secrets.get("CFBD_API_KEY", "")

    if not api_key:
        st.warning("CFBD_API_KEY is missing from Streamlit Secrets.")
        return None

    try:
        response = requests.get(
            f"{CFBD_BASE_URL}/teams/fbs",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        names = {
            item["school"]
            for item in payload
            if isinstance(item, dict) and item.get("school")
        }

        if not names:
            st.warning("The API returned no FBS team names.")
            return None

        return names

    except requests.exceptions.RequestException as error:
        st.warning(
            f"Could not retrieve FBS teams from CollegeFootballData "
            f"({type(error).__name__}). Check your API key and API access."
        )
        return None

    except (ValueError, TypeError) as error:
        st.warning(
            f"Could not read the FBS team response ({type(error).__name__})."
        )
        return None


def get_fcs_teams():
    api_key = st.secrets.get("CFBD_API_KEY", "")
    if not api_key:
        return set()

    try:
        response = requests.get(
            f"{CFBD_BASE_URL}/teams/fbs",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        fbs_payload = response.json()

        # Get all teams and identify those not classified as FBS.
        response = requests.get(
            f"{CFBD_BASE_URL}/teams",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        all_teams = response.json()

        fbs_names = {
            team["school"].strip().lower()
            for team in fbs_payload
            if isinstance(team, dict) and team.get("school")
        }

        return {
            team["school"].strip().lower()
            for team in all_teams
            if isinstance(team, dict)
            and team.get("school")
            and team["school"].strip().lower() not in fbs_names
        }

    except (requests.exceptions.RequestException, ValueError, TypeError):
        return set()


@st.cache_data(ttl=900)
def get_cfb_odds():
    api_key = st.secrets.get("ODDS_API_KEY", "")

    if not api_key:
        return {}

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals,spreads",
        "oddsFormat": "american",
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        games = response.json()

        odds_by_matchup = {}

        for game in games:
            home = game.get("home_team", "").strip().lower()
            away = game.get("away_team", "").strip().lower()

            totals = []
            spreads = []

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                totals.append(outcome.get("point"))

                    elif market.get("key") == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home:
                                spreads.append(outcome.get("point"))

            odds_by_matchup[(away, home)] = {
                "total": (
                    sum(totals) / len(totals) if totals else None
                ),
                "home_spread": (
                    sum(spreads) / len(spreads) if spreads else None
                ),
            }

        return odds_by_matchup

    except (requests.exceptions.RequestException, ValueError):
        return {}


from datetime import datetime


def fetch_current_season_games():
    api_key = st.secrets.get("CFBD_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "CFBD_API_KEY is missing from Streamlit Secrets."
        )

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
            f"CollegeFootballData returned HTTP {response.status_code}."
        )

    payload = response.json()

    if not payload:
        raise ValueError(f"No games were returned for {season}.")

    live = pd.DataFrame(payload)

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
            "API response missing fields: " + ", ".join(missing)
        )

    live["home_points"] = pd.to_numeric(
        live["home_points"], errors="coerce"
    )
    live["away_points"] = pd.to_numeric(
        live["away_points"], errors="coerce"
    )

    completed = live.dropna(
        subset=["home_points", "away_points"]
    ).copy()

    upcoming = live[
        live["home_points"].isna() | live["away_points"].isna()
    ].copy()

    return completed, upcoming

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
st.title("Dash's Model 🏈")
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


# Automatically fetch current-season games when the app first loads.
if "live_completed" not in st.session_state or "live_upcoming" not in st.session_state:
    try:
        with st.spinner("Loading current-season games..."):
            live_completed, live_upcoming = fetch_current_season_games()

            st.session_state["live_completed"] = live_completed
            st.session_state["live_upcoming"] = live_upcoming
            st.session_state["live_fetch_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    except Exception as error:
        st.error(f"Could not fetch live games: {error}")

# Keep a manual refresh button as a backup.
if st.button("Refresh current season games"):
    try:
        with st.spinner("Refreshing games from CollegeFootballData..."):
            live_completed, live_upcoming = fetch_current_season_games()

            st.session_state["live_completed"] = live_completed
            st.session_state["live_upcoming"] = live_upcoming
            st.session_state["live_fetch_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    except Exception as error:
        st.error(f"Could not refresh live games: {error}")



if "live_completed" in st.session_state:
    live_completed = st.session_state["live_completed"].copy()
    live_upcoming = st.session_state["live_upcoming"].copy()

    # Make sure live results have a season value.
    current_season = datetime.now().year

    if "season" not in live_completed.columns:
        live_completed["season"] = current_season

    live_completed["season"] = (
        pd.to_numeric(live_completed["season"], errors="coerce")
        .fillna(current_season)
        .astype(int)
    )

    games["season"] = pd.to_numeric(
        games["season"], errors="coerce"
    ).astype("Int64")

    live_completed["season"] = live_completed["season"].astype("Int64")

    # Combine historical games and current-season final scores.
    combined = pd.concat(
        [games, live_completed],
        ignore_index=True,
        sort=False,
    )

    # Normalize game IDs so numeric/string representations match.
    if "id" in combined.columns:
        combined["id"] = combined["id"].astype("string")

        with_id = combined[combined["id"].notna()]
        without_id = combined[combined["id"].isna()]

        with_id = with_id.drop_duplicates(
            subset=["id"], keep="last"
        )

        combined = pd.concat(
            [with_id, without_id],
            ignore_index=True,
            sort=False,
        )

    # Restore chronological order before calculating Elo ratings.
    if "start_date" in combined.columns:
        combined["start_date"] = pd.to_datetime(
            combined["start_date"], errors="coerce", utc=True
        )
        combined = combined.sort_values(
            ["start_date", "season"], na_position="last"
        )

    games = combined.reset_index(drop=True)

    # Show the number of games used for the current season.
    season_games = games[games["season"] == current_season]
    season_completed = season_games.dropna(
        subset=["home_points", "away_points"]
    )

    st.success(
        f"Live data loaded. {len(season_completed)} games with scores "
        f"are included for {current_season}; "
        f"{len(live_upcoming)} scheduled games remain."
    )

    st.caption(
        "Games are combined with historical data and duplicate IDs "
        "are removed before ratings are recalculated."
    )

    if not live_upcoming.empty:
        
        st.subheader("Upcoming Games")

        rankings, _, _ = calculate_ratings(games)

        # Create a lookup table of team ratings.
        rating_lookup = rankings.set_index("Team")["Rating"].to_dict()

        upcoming_display = live_upcoming.copy()


        # Sort games by kickoff time and display in New York time.
        if "start_date" in upcoming_display.columns:
            upcoming_display["start_date"] = pd.to_datetime(
                upcoming_display["start_date"], errors="coerce", utc=True
            )
        
            upcoming_display = upcoming_display.sort_values(
                "start_date", na_position="last"
            )
        
            upcoming_display["start_date"] = (
                upcoming_display["start_date"]
                .dt.tz_convert("America/New_York")
                .dt.strftime("%a, %b %-d • %-I:%M %p ET")
            )
        
        # Filter upcoming games by week.
        if not upcoming_display.empty and "week" in upcoming_display.columns:
            available_weeks = sorted(
                upcoming_display["week"].dropna().unique()
            )

            if available_weeks:
                default_week = available_weeks[0]

                selected_week = st.selectbox(
                    "Select Week",
                    options=available_weeks,
                    index=0,
                    format_func=lambda week: f"Week {int(week)}",
                    key="upcoming_week_selector",
                )

                upcoming_display = upcoming_display[
                    upcoming_display["week"] == selected_week
                ].copy()



        # Calculate a prediction for every scheduled game.
        fcs_teams = get_fcs_teams() or set()
        game_odds = get_cfb_odds()
        
        
        def predict_game(row, fcs_teams, game_odds):
            home = str(row["home_team"]).strip()
            away = str(row["away_team"]).strip()

            FCS_BASE_RATING = 1250

            
            home_key = home.strip().lower()
            away_key = away.strip().lower()

            FCS_BASE_RATING = 1250

            home_rating = rating_lookup.get(home, START_RATING)
            away_rating = rating_lookup.get(away, START_RATING)

            if home_key in fcs_teams:
                home_rating = FCS_BASE_RATING

            if away_key in fcs_teams:
                away_rating = FCS_BASE_RATING


            neutral = bool(row.get("neutral_site", False))
            home_edge = 0 if neutral else HOME_ADVANTAGE_ELO

            # Expected home-team win probability from Elo ratings.
            rating_diff = home_rating + home_edge - away_rating
            home_prob = 1 / (1 + 10 ** (-rating_diff / 400))

            # Convert the rating difference to a rough projected margin.
            projected_margin = rating_diff / 20

            # Estimate total points using a simple baseline.
            # Use the betting market's projected total when available.
            odds_key = (away.strip().lower(), home.strip().lower())
            market_total = game_odds.get(odds_key, {}).get("total")

            projected_total = (
                market_total if market_total is not None else 48
            )
            home_score = round((projected_total + projected_margin) / 2)
            away_score = round((projected_total - projected_margin) / 2)

            return pd.Series({
                "Predicted Away Score": max(0, away_score),
                "Predicted Home Score": max(0, home_score),
                "Projected Spread (Home)": (
    f"{home} -{abs(projected_margin):.1f}"
    if projected_margin >= 0
    else f"{away} -{abs(projected_margin):.1f}"
),
                "Home Win Probability": f"{home_prob:.1%}",
                "Predicted Winner": home if home_prob >= 0.5 else away,
            })

        predictions = upcoming_display.apply(
            lambda row: predict_game(row, fcs_teams, game_odds),
            axis=1
        )
        upcoming_display = pd.concat(
            [upcoming_display, predictions], axis=1
        )

        # Show the most useful columns first.
        
        preferred_columns = [
            "start_date",
            "week",
            "away_team",
            "Predicted Away Score",
            "home_team",
            "Predicted Home Score",
            "Predicted Winner",
            "Projected Spread (Home)",
            "Home Win Probability",
        ]

        display_names = {
            "start_date": "Kickoff (Eastern)",
            "week": "Week",
            "away_team": "Away Team",
            "Predicted Away Score": "Away Score",
            "home_team": "Home Team",
            "Predicted Home Score": "Home Score",
            "Predicted Winner": "Predicted Winner",
            "Projected Spread (Home)": "Projected Spread",
            "Home Win Probability": "Home Win Probability",
        }


        display_columns = [
            col for col in preferred_columns
            if col in upcoming_display.columns
        ]

        
        
        
        
        
        # Build a table with centered headers and cell contents.
        table_data = (
            upcoming_display[display_columns]
            .rename(columns=display_names)
            .reset_index(drop=True)
        )
        
        # ESPN logo URLs for college football teams.
        
        # ESPN logo URLs using ESPN's college football team identifiers.
        
        
        
        @st.cache_data(ttl=86400)
        def get_team_logo_map():
            try:
                response = requests.get(
                    "https://site.api.espn.com/apis/site/v2/sports/"
                    "football/college-football/teams",
                    params={"limit": 500},
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()

                logo_map = {}

                for item in payload.get("sports", []):
                    for league in item.get("leagues", []):
                        for entry in league.get("teams", []):
                            team = entry.get("team", {})
                            logos = team.get("logos", [])
                            logo_url = (
                                logos[0].get("href", "")
                                if logos else ""
                            )

                            if not logo_url:
                                continue

                            names = [
                                team.get("displayName"),
                                team.get("name"),
                                team.get("shortDisplayName"),
                                team.get("abbreviation"),
                                team.get("location"),
                                team.get("nickname"),
                            ]

                            for name in names:
                                if name:
                                    logo_map[name.strip().lower()] = logo_url

                            team_id = team.get("id")
                            if team_id:
                                logo_map[f"id:{team_id}"] = logo_url

                return logo_map

            except (requests.exceptions.RequestException, ValueError):
                return {}


            except (requests.exceptions.RequestException, ValueError):
                return {}



        header_html = "".join(
            f'<th style="text-align:center; padding:8px; '
            f'border-bottom:1px solid #888;">{column}</th>'
            for column in table_data.columns
        )

        
        rows_html = ""
        logo_map = get_team_logo_map()

        for _, row in table_data.iterrows():
            cells_html = ""

            for column, value in row.items():
                safe_value = html.escape(str(value))

                
                
                
                if column in ("Away Team", "Home Team"):
                    team_key = str(value).strip().lower()
                    original_team_key = team_key

                    team_aliases = {
                        "utsa": "utsa roadrunners",
                        "southern miss": "southern miss golden eagles",
                        "smu": "smu mustangs",
                        "nebraska": "nebraska cornhuskers",
                        "illinois": "illinois fighting illini",
                        "new mexico": "new mexico lobos",
                        "tcu": "tcu horned frogs",
                        "arizona": "arizona wildcats",
                        "minnesota": "minnesota golden gophers",
                        "missouri": "missouri tigers",
                        "texas": "texas longhorns",
                        "north carolina": "north carolina tar heels",
                        "baylor": "baylor bears",
                        "tulane": "tulane green wave",
                        "utah": "utah utes",
                        "purdue": "purdue boilermakers",
                        "duke": "duke blue devils",
                        "arkansas": "arkansas razorbacks",
                        "oklahoma state": "oklahoma state cowboys",
                        "kansas state": "kansas state wildcats",
                        "south carolina": "south carolina gamecocks",
                        "delaware": "delaware fightin blue hens",
                        "washington": "washington huskies",
                        "merrimack": "merrimack warriors",
                        "marshall": "marshall thundering herd",
                        "alabama": "alabama crimson tide",
                        "houston": "houston cougars",
                        "oregon": "oregon ducks",
                        "kansas": "kansas jayhawks",
                        "troy": "troy trojans",
                        "wisconsin": "wisconsin badgers",
                        "louisiana": "louisiana ragin cajuns",
                        "kentucky": "kentucky wildcats",
                        "tennessee": "tennessee volunteers",
                        "sam houston": "sam houston bearkats",
                        "miami": "miami hurricanes",
                        "colorado": "colorado buffaloes",
                        "iowa": "iowa hawkeyes",
                        "northwestern": "northwestern wildcats",
                        "georgia": "georgia bulldogs",
                        "michigan": "michigan wolverines",
                        "oklahoma": "oklahoma sooners",
                        "florida": "florida gators",
                        "west virginia": "west virginia mountaineers",
                        "texas tech": "texas tech red raiders",
                        "utep": "utep miners",
                        "wyoming": "wyoming cowboys",
                        "app state": "appalachian state mountaineers",
                        "miami (fl)": "miami hurricanes",
                        "miami (oh)": "miami redhawks",
                        "ole miss": "ole miss rebels",
                        "ucf": "ucf knights",
                        "lsu": "lsu tigers",
                        "usc": "usc trojans",
                        "ucla": "ucla bruins",
                    }

                    
                    team_key = original_team_key
                    
                    logo_url = logo_map.get(team_key, "")

                    # Normalize team names and match ESPN's available names.
                    def normalize_team_name(name):
                        return "".join(
                            char for char in str(name).lower()
                            if char.isalnum()
                        )

                    if not logo_url:
                        normalized_key = normalize_team_name(team_key)

                        matches = [
                            url for name, url in logo_map.items()
                            if normalize_team_name(name) == normalized_key
                            or normalized_key in normalize_team_name(name)
                        ]

                        # Use a partial match only when it identifies one logo.
                        if len(matches) == 1:
                            logo_url = matches[0]


                        

                    if logo_url:
                        cell_content = (
                            f'<img src="{logo_url}" '
                            f'alt="" width="24" height="24" '
                            f'style="object-fit:contain; '
                            f'vertical-align:middle; margin-right:6px;">'
                            f'{safe_value}'
                        )
                    else:
                        cell_content = safe_value
                else:
                    cell_content = safe_value


                cells_html += (
                    f'<td style="text-align:center; padding:8px; '
                    f'border-bottom:1px solid #888;">'
                    f'{cell_content}</td>'
                )

            rows_html += f"<tr>{cells_html}</tr>"


        table_html = f"""
        <div style="width:100%; overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse;">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """

        st.html(table_html)








rankings, _, _ = calculate_ratings(games)


if rankings.empty:
    st.error("No valid games were found in the CSV.")
    st.stop()

fbs_teams = get_fbs_teams()
fcs_teams = get_fcs_teams()


st.caption(f"Odds API: loaded odds for {len(game_odds)} games.")

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
