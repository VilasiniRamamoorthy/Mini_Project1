
import ast
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# ============================================================

st.set_page_config(
    page_title="TMDB Movie Analytics",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "tmdb_movies.db"

# GitHub raw CSV URLs

GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "VilasiniRamamoorthy/Mini_Project1/main/"
)

CSV_FILES = {
    "movies": GITHUB_RAW_URL + "movies.csv",
    "cast": GITHUB_RAW_URL + "cast.csv",
    "crew": GITHUB_RAW_URL + "crew.csv",
    "genres": GITHUB_RAW_URL + "genres.csv",
    "movie_genres": GITHUB_RAW_URL + "movie_genres.csv",
    "movie_keywords": GITHUB_RAW_URL + "movie_keywords.csv",
}


# ============================================================
# 2. COMMON HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """Find a column without depending on capitalization."""

    lookup = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in lookup:
            return lookup[key]

    return None


def normalize_id(value):
    """Normalize IDs so values such as 123 and 123.0 match."""

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text if text else None


def parse_list(value):
    """Convert list-like CSV values into Python lists."""

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]

    if value is None:
        return []

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {
        "nan", "none", "null", "[]"
    }:
        return []

    if text.startswith(("[", "{")):

        for parser in (json.loads, ast.literal_eval):

            try:
                parsed = parser(text)

                if isinstance(parsed, dict):
                    parsed = [parsed]

                if isinstance(parsed, list):

                    output = []

                    for item in parsed:

                        if isinstance(item, dict):

                            name = (
                                item.get("name")
                                or item.get("original_name")
                                or item.get("title")
                            )

                            if name:
                                output.append(str(name).strip())

                        elif item is not None:
                            output.append(str(item).strip())

                    if output:
                        return output

            except (ValueError, SyntaxError, TypeError):
                pass

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


# ============================================================
# 3. LOAD ALL SIX CSV FILES
# ============================================================

@st.cache_data(show_spinner="Loading all six TMDB CSV files...")
@st.cache_data
def load_all_csv_files():

    data = {}

    for file_name, file_url in CSV_FILES.items():

        try:
            data[file_name] = pd.read_csv(file_url)
            data[file_name].columns = (
                data[file_name].columns.str.strip()
            )

        except Exception as e:
            st.error(f"Error loading {file_name}.csv: {e}")
            return None

    return data

# ============================================================
# 4. BUILD GENRE MAPPING
# ============================================================

def build_genre_mapping(genres_df, movie_genres_df):

    genre_lookup = genres_df[
        ["genre_id", "genre_name"]
    ].copy()

    movie_genre_links = movie_genres_df[
        ["movie_id", "genre_id"]
    ].copy()

    genre_lookup["genre_id"] = (
        genre_lookup["genre_id"].apply(normalize_id)
    )

    movie_genre_links["genre_id"] = (
        movie_genre_links["genre_id"].apply(normalize_id)
    )

    movie_genre_links["movie_id"] = (
        movie_genre_links["movie_id"].apply(normalize_id)
    )

    genre_lookup["genre_name"] = (
        genre_lookup["genre_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    merged = movie_genre_links.merge(
        genre_lookup,
        on="genre_id",
        how="left"
    )

    merged = merged.dropna(subset=["genre_name"])

    merged = merged[
        merged["genre_name"].astype(str).str.strip() != ""
    ]

    genre_mapping = (
        merged.groupby("movie_id")["genre_name"]
        .apply(
            lambda values: sorted(
                set(values.astype(str).str.strip())
            )
        )
        .to_dict()
    )

    return genre_mapping


# ============================================================
# 5. BUILD CAST MAPPING
# ============================================================

def build_cast_mapping(cast_df):
    """
    Build a mapping using:
    cast.csv movie_id -> actor_name

    One movie can have multiple actors.
    """

    cast_df = cast_df.copy()

    required_columns = ["movie_id", "actor_name"]

    for col in required_columns:
        if col not in cast_df.columns:
            raise ValueError(
                f"{col} column is missing in cast.csv."
            )

    # Normalize movie IDs
    cast_df["movie_id"] = (
        cast_df["movie_id"].apply(normalize_id)
    )

    # Clean actor names
    cast_df["actor_name"] = (
        cast_df["actor_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove missing movie IDs and actor names
    cast_df = cast_df.dropna(
        subset=["movie_id"]
    )

    cast_df = cast_df[
        cast_df["actor_name"] != ""
    ]

    cast_df = cast_df[
        cast_df["actor_name"].str.lower() != "nan"
    ]

    # Sort by cast order if available
    if "cast_order" in cast_df.columns:

        cast_df["cast_order"] = pd.to_numeric(
            cast_df["cast_order"],
            errors="coerce"
        )

        cast_df = cast_df.sort_values(
            "cast_order",
            na_position="last"
        )

    # Remove duplicate actor-movie combinations
    cast_df = cast_df.drop_duplicates(
        subset=["movie_id", "actor_name"]
    )

    # Create movie_id -> actor names mapping
    cast_mapping = (
        cast_df.groupby("movie_id")["actor_name"]
        .apply(list)
        .to_dict()
    )

    return cast_mapping


# ============================================================
# 6. PREPARE MOVIE DATA USING THE SIX CSV FILES
# ============================================================

def prepare_data(data):

    movies_df = data["movies"].copy()

    movie_id_column = find_column(
        movies_df,
        ["movie_id"]
    )

    if movie_id_column is None:
        raise ValueError(
            "movie_id column is missing in movies.csv."
        )

    # Standardize movie ID column
    movies_df["movie_id"] = movies_df[
        movie_id_column
    ]

    movies_df["_movie_id_key"] = (
        movies_df["movie_id"].apply(normalize_id)
    )

    # --------------------------------------------------------
    # Build genre mapping
    # --------------------------------------------------------

    genre_mapping = build_genre_mapping(
        data["genres"],
        data["movie_genres"]
    )

    movies_df["genre"] = (
        movies_df["_movie_id_key"].map(genre_mapping)
    )

    movies_df["genre"] = movies_df["genre"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    # --------------------------------------------------------
    # Build actor mapping using actor_name from cast.csv
    # --------------------------------------------------------

    cast_mapping = build_cast_mapping(
        data["cast"]
    )

    movies_df["cast"] = (
        movies_df["_movie_id_key"].map(cast_mapping)
    )

    movies_df["cast"] = movies_df["cast"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    df = movies_df.copy()

    # --------------------------------------------------------
    # Standardize commonly used movie columns
    # --------------------------------------------------------

    aliases = {
        "title": [
            "title",
            "original_title",
            "movie_title",
            "name"
        ],
        "budget": ["budget"],
        "revenue": ["revenue", "box_office"],
        "runtime": [
            "runtime",
            "runtime_minutes",
            "duration"
        ],
        "popularity": ["popularity"],
        "vote_average": [
            "vote_average",
            "rating",
            "average_rating"
        ],
        "vote_count": [
            "vote_count",
            "num_votes"
        ],
        "release_date": [
            "release_date",
            "released",
            "release_year",
            "year"
        ],
        "language": [
            "original_language",
            "language",
            "lang"
        ],
    }

    for target, candidates in aliases.items():

        source_column = find_column(
            df,
            candidates
        )

        if source_column is not None and source_column != target:
            df[target] = df[source_column]

    if "title" not in df:
        df["title"] = [
            f"Movie {i + 1}"
            for i in range(len(df))
        ]

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count"
    ]

    for col in numeric_columns:

        if col not in df:
            df[col] = pd.NA

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Extract release year
    # --------------------------------------------------------

    if "release_date" not in df:
        df["release_date"] = pd.NA

    release_text = df["release_date"].astype("string")

    extracted_year = release_text.str.extract(
        r"((?:19|20)\d{2})",
        expand=False
    )

    df["release_year"] = pd.to_numeric(
        extracted_year,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Genre columns
    # --------------------------------------------------------

    df["genre_list"] = df["genre"].apply(parse_list)

    df["primary_genre"] = df["genre_list"].apply(
        lambda values: values[0] if values else "Unknown"
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    if "language" not in df:
        df["language"] = "Unknown"

    df["language"] = (
        df["language"]
        .fillna("Unknown")
        .astype(str)
        .replace("", "Unknown")
    )

    # --------------------------------------------------------
    # Actor list
    # Uses actor_name from cast.csv
    # --------------------------------------------------------

    df["actor_list"] = df["cast"].apply(
        lambda value: (
            value if isinstance(value, list) else []
        )
    )

    # Add row identifier
    df = df.reset_index(drop=True)

    df["_movie_row_id"] = df.index

    return df


# ============================================================
# 7. FILTER STATE
# ============================================================

def seed_filter_state(df, genres):

    years = df["release_year"].dropna()

    min_year = int(years.min()) if not years.empty else 1900
    max_year = int(years.max()) if not years.empty else 2026

    languages = sorted(
        df["language"].dropna().astype(str).unique().tolist()
    )

    if "filter_genres" not in st.session_state:
        st.session_state["filter_genres"] = []

    if "filter_years" not in st.session_state:
        st.session_state["filter_years"] = (min_year, max_year)

    if "filter_min_rating" not in st.session_state:
        st.session_state["filter_min_rating"] = 0.0

    if "filter_language" not in st.session_state:
        st.session_state["filter_language"] = "All languages"

    return genres, min_year, max_year, languages

# ============================================================
# 8. RESET FILTERS
# ============================================================

def reset_filters(min_year, max_year):

    st.session_state["filter_genres"] = []

    st.session_state["filter_years"] = (
        min_year,
        max_year
    )

    st.session_state["filter_min_rating"] = 0.0

    st.session_state["filter_language"] = "All languages"

# ============================================================
# 8A. SHARED SIDEBAR FILTERS
# ============================================================

def render_sidebar_filters(genres, min_year, max_year, languages):

    st.sidebar.subheader("🎛️ Movie Filters")

    # Genre multiselect
    st.sidebar.multiselect(
        "Select Genre",
        options=genres,
        key="filter_genres"
    )

    # Release year range slider
    st.sidebar.slider(
        "Release Year Range",
        min_value=min_year,
        max_value=max_year,
        key="filter_years"
    )

    # Minimum rating slider
    st.sidebar.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        key="filter_min_rating"
    )

    # Language selectbox
    language_options = ["All languages"] + languages

    if st.session_state["filter_language"] not in language_options:
        st.session_state["filter_language"] = "All languages"

    st.sidebar.selectbox(
        "Select Language",
        options=language_options,
        key="filter_language"
    )

    # Reset button
    st.sidebar.button(
        "🔄 Reset All Filters",
        on_click=reset_filters,
        args=(min_year, max_year),
        use_container_width=True
    )


# ============================================================
# 9. APPLY FILTERS
# ============================================================

def apply_filters(df):

    filtered = df.copy()

    selected_genres = st.session_state.get(
        "filter_genres", []
    )

    year_range = st.session_state.get(
        "filter_years", None
    )

    min_rating = st.session_state.get(
        "filter_min_rating", 0.0
    )

    language = st.session_state.get(
        "filter_language", "All languages"
    )

    # Genre filter
    if selected_genres:

        filtered = filtered[
            filtered["genre_list"].apply(
                lambda values: any(
                    genre in selected_genres
                    for genre in values
                )
            )
        ]

    # Release year filter
    if year_range:

        low, high = year_range

        filtered = filtered[
            filtered["release_year"].notna()
            & filtered["release_year"].between(low, high)
        ]

    # Rating filter
    if min_rating > 0:

        filtered = filtered[
            filtered["vote_average"].notna()
            & (
                filtered["vote_average"] >= min_rating
            )
        ]

    # Language filter
    if language != "All languages":

        filtered = filtered[
            filtered["language"] == language
        ]

    return filtered


# ============================================================
# 10. FILTERS PAGE
# ============================================================

def render_filters(df, genres, min_year, max_year, languages):

    st.title("🎛️ Movie Filters")

    st.caption(
        "Choose filters below. These filters are shared "
        "across Dashboard and Query Explorer."
    )

    with st.container(border=True):

        st.subheader("Genre")

        st.multiselect(
            "Select Genre",
            options=genres,
            key="filter_genres",
            help="Select one or more movie genres."
        )

        st.subheader("Release Year")

        st.slider(
            "Release Year Range",
            min_value=min_year,
            max_value=max_year,
            key="filter_years"
        )

        st.subheader("Minimum Rating")

        st.slider(
            "Minimum Rating",
            min_value=0.0,
            max_value=10.0,
            step=0.5,
            key="filter_min_rating"
        )

        st.subheader("Original Language")

        language_options = (
            ["All languages"] + languages
        )

        if (
            st.session_state["filter_language"]
            not in language_options
        ):
            st.session_state["filter_language"] = "All languages"

        st.selectbox(
            "Select Language",
            options=language_options,
            key="filter_language"
        )

        st.button(
            "🔄 Reset All Filters",
            on_click=reset_filters,
            args=(min_year, max_year),
            type="secondary"
        )

    filtered = apply_filters(df)

    st.metric(
        "🎬 Movies Matching Current Filters",
        f"{len(filtered):,}"
    )

    st.subheader("Filtered Movie Preview")

    st.dataframe(
        filtered[
            [
                "title",
                "release_year",
                "primary_genre",
                "vote_average",
                "language"
            ]
        ].head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. GENRE COUNTS
# ============================================================

def genre_counts(filtered):

    exploded = filtered[
        ["genre_list"]
    ].explode("genre_list")

    exploded = exploded.dropna(
        subset=["genre_list"]
    )

    exploded = exploded[
        exploded["genre_list"].astype(str).str.strip() != ""
    ]

    return (
        exploded["genre_list"]
        .value_counts()
        .rename_axis("genre")
        .reset_index(name="movie_count")
    )


# ============================================================
# 12. DASHBOARD
# ============================================================

def render_dashboard(filtered):

    st.title("🎬 TMDB Movie Dashboard")

    st.caption(
        f"Showing {len(filtered):,} movies "
        "matching the selected filters."
    )

    genre_table = genre_counts(filtered)

    total_genres = (
        int(genre_table["genre"].nunique())
        if not genre_table.empty else 0
    )

    actor_set = {
        actor
        for actors in filtered["actor_list"]
        for actor in actors
    }

    avg_rating = filtered["vote_average"].mean()

    # KPI metrics
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Movies",
        f"{len(filtered):,}"
    )

    k2.metric(
        "Genres Represented",
        f"{total_genres:,}"
    )

    k3.metric(
        "Unique Actors",
        f"{len(actor_set):,}"
    )

    k4.metric(
        "Average Rating",
        f"{avg_rating:.2f}"
        if pd.notna(avg_rating) else "N/A"
    )

    st.divider()

    # Top 10 movies by revenue
    st.subheader("🏆 Top 10 Movies by Revenue")

    top_revenue = (
        filtered.dropna(subset=["revenue"])
        .nlargest(10, "revenue")
    )

    if top_revenue.empty:

        st.info(
            "No revenue values available for the current filters."
        )

    else:

        fig = px.bar(
            top_revenue.sort_values("revenue"),
            x="revenue",
            y="title",
            orientation="h",
            labels={
                "revenue": "Revenue",
                "title": "Movie"
            },
            title="Highest-Grossing Movies",
            hover_data=[
                "budget",
                "vote_average",
                "popularity"
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    left, right = st.columns(2)

    # Genre pie chart
    with left:

        st.subheader("Genre Distribution")

        if genre_table.empty:

            st.info("No genre data available.")

        else:

            fig = px.pie(
                genre_table,
                names="genre",
                values="movie_count",
                hole=0.42
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Budget vs revenue scatter plot
    with right:

        st.subheader("Budget vs Revenue")

        scatter_data = filtered.dropna(
            subset=["budget", "revenue"]
        )

        scatter_data = scatter_data[
            (scatter_data["budget"] > 0)
            & (scatter_data["revenue"] >= 0)
        ]

        if scatter_data.empty:

            st.info(
                "No valid budget and revenue pairs available."
            )

        else:

            fig = px.scatter(
                scatter_data,
                x="budget",
                y="revenue",
                color="primary_genre",
                hover_name="title",
                hover_data=[
                    "vote_average",
                    "popularity"
                ],
                labels={
                    "budget": "Budget",
                    "revenue": "Revenue",
                    "primary_genre": "Primary Genre"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Automatic insights
    st.subheader("💡 Automatic Insights")

    if not genre_table.empty:

        top_genre = genre_table.iloc[0]

        insight = (
            f"The most represented genre is "
            f"**{top_genre['genre']}**, with "
            f"{int(top_genre['movie_count']):,} "
            "genre assignments. "
        )

    else:

        insight = (
            "Genre information is unavailable "
            "for the current selection. "
        )

    if not top_revenue.empty:

        best = top_revenue.iloc[0]

        insight += (
            f"The highest-revenue movie is "
            f"**{best['title']}**, with revenue "
            f"{best['revenue']:,.0f}. "
        )

    if pd.notna(avg_rating):

        insight += (
            f"The average vote rating is "
            f"**{avg_rating:.2f}/10**."
        )

    st.info(insight)


# ============================================================
# 13. SAVED SQL QUERIES
# ============================================================

SAVED_QUERIES = {

    "Top 10 Movies by Revenue": {
        "sql": """
            SELECT title, budget, revenue,
                   vote_average, popularity
            FROM filtered_movies
            WHERE revenue IS NOT NULL
            ORDER BY revenue DESC
            LIMIT 10
        """,
        "chart": "bar",
        "x": "revenue",
        "y": "title",
        "title": "Top 10 Movies by Revenue",
        "insight": (
            "These are the highest-revenue movies "
            "among the currently filtered records."
        )
    },

    "Average Rating by Primary Genre": {
        "sql": """
            SELECT primary_genre,
                   ROUND(AVG(vote_average), 2) AS average_rating,
                   COUNT(*) AS movie_count
            FROM filtered_movies
            WHERE vote_average IS NOT NULL
            GROUP BY primary_genre
            ORDER BY average_rating DESC
        """,
        "chart": "bar",
        "x": "primary_genre",
        "y": "average_rating",
        "title": "Average Rating by Primary Genre",
        "insight": (
            "Compare average audience ratings by primary genre. "
            "Genres with fewer movies may have less stable averages."
        )
    },

    "Movie Count by Release Year": {
        "sql": """
            SELECT release_year,
                   COUNT(*) AS movie_count
            FROM filtered_movies
            WHERE release_year IS NOT NULL
            GROUP BY release_year
            ORDER BY release_year
        """,
        "chart": "line",
        "x": "release_year",
        "y": "movie_count",
        "title": "Movie Count by Release Year",
        "insight": (
            "This trend shows how the number of movies "
            "changes across release years."
        )
    },

    "Budget vs Revenue": {
        "sql": """
            SELECT title, budget, revenue,
                   popularity, vote_average
            FROM filtered_movies
            WHERE budget > 0 AND revenue >= 0
            ORDER BY revenue DESC
        """,
        "chart": "scatter",
        "x": "budget",
        "y": "revenue",
        "title": "Budget vs Revenue",
        "insight": (
            "A larger movie budget does not guarantee "
            "higher revenue. Look for patterns and outliers."
        )
    },

    "Top 10 Prolific Actor": {
        "sql": """
            SELECT
                actor_name,
                COUNT(DISTINCT movie_id) AS movie_count,
                ROUND(AVG(vote_average), 2) AS average_rating
            FROM actor_movie_appearances
            WHERE vote_average IS NOT NULL
            GROUP BY actor_name
            ORDER BY movie_count DESC, average_rating DESC
            LIMIT 10
        """,
        "chart": "actor_bar",
        "x": "movie_count",
        "y": "actor_name",
        "title": "Top 10 Prolific Actors by Movie Count",
        "insight": (
            "These are the 10 actors who appeared in the highest "
            "number of movies in the currently filtered selection. "
            "The average rating indicates the audience rating of "
            "their movies. A higher movie count indicates greater "
            "screen presence in this dataset, while average rating "
            "provides a view of the ratings of those movies."
        )
    },

    "Top Rated Movies": {
        "sql": """
            SELECT title, vote_average, vote_count
            FROM filtered_movies
            WHERE vote_average IS NOT NULL
            ORDER BY vote_average DESC, vote_count DESC
            LIMIT 10
        """,
        "chart": "bar",
        "x": "vote_average",
        "y": "title",
        "title": "Top Rated Movies",
        "insight": (
            "These movies have the highest audience ratings "
            "within the current filter selection."
        )
    }
}


# ============================================================
# 14. QUERY EXPLORER
# ============================================================

def render_query_explorer(filtered):

    st.title("🔎 SQL Query Explorer")

    st.caption(
        "Run saved SQL queries against the movies "
        "that match your selected filters."
    )

    selected_query = st.selectbox(
        "Choose a Saved SQL Query",
        list(SAVED_QUERIES.keys())
    )

    query_info = SAVED_QUERIES[selected_query]

    with st.expander("View SQL Query"):

        st.code(
            query_info["sql"],
            language="sql"
        )

    # Create temporary SQL database
    with sqlite3.connect(":memory:") as conn:

        sql_df = filtered.copy()

        # Convert list columns into SQL-friendly strings
        sql_df["genre_list"] = sql_df["genre_list"].apply(
            lambda values: (
                ", ".join(values)
                if isinstance(values, list)
                else ""
            )
        )

        sql_df["actor_list"] = sql_df["actor_list"].apply(
            lambda values: (
                ", ".join(values)
                if isinstance(values, list)
                else ""
            )
        )

        # Create filtered_movies SQL table
        sql_df.to_sql(
            "filtered_movies",
            conn,
            index=False,
            if_exists="replace"
        )

        # ----------------------------------------------------
        # Create actor_movie_appearances SQL table
        # ----------------------------------------------------

        actor_movie_df = filtered[
            ["movie_id", "actor_list", "vote_average"]
        ].copy()

        # Normalize movie IDs
        actor_movie_df["movie_id"] = (
            actor_movie_df["movie_id"].apply(normalize_id)
        )

        # Convert each actor into a separate row
        actor_movie_df = actor_movie_df.explode(
            "actor_list"
        )

        # Rename actor_list to actor_name
        actor_movie_df = actor_movie_df.rename(
            columns={
                "actor_list": "actor_name"
            }
        )

        # Clean actor names
        actor_movie_df["actor_name"] = (
            actor_movie_df["actor_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Remove empty actor names and missing movie IDs
        actor_movie_df = actor_movie_df.dropna(
            subset=["movie_id"]
        )

        actor_movie_df = actor_movie_df[
            actor_movie_df["actor_name"] != ""
        ]

        actor_movie_df = actor_movie_df[
            actor_movie_df["actor_name"].str.lower() != "nan"
        ]

        # Avoid duplicate actor-movie combinations
        actor_movie_df = actor_movie_df.drop_duplicates(
            subset=["movie_id", "actor_name"]
        )

        # Store actor data in SQLite
        actor_movie_df.to_sql(
            "actor_movie_appearances",
            conn,
            index=False,
            if_exists="replace"
        )

        # ----------------------------------------------------
        # Execute selected SQL query
        # ----------------------------------------------------

        try:

            result = pd.read_sql_query(
                query_info["sql"],
                conn
            )

        except Exception as exc:

            st.error(
                f"Could not execute SQL query: {exc}"
            )

            return

    # --------------------------------------------------------
    # Display SQL results
    # --------------------------------------------------------

    st.subheader("Query Results")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Display appropriate chart
    # --------------------------------------------------------

    if not result.empty:

        chart_type = query_info["chart"]

        if chart_type == "actor_bar":

            fig = px.bar(
                result.sort_values("movie_count"),
                x="movie_count",
                y="actor_name",
                orientation="h",
                title=query_info["title"],
                labels={
                    "movie_count": "Number of Movies",
                    "actor_name": "Actor",
                    "average_rating": "Average Movie Rating"
                },
                hover_data={
                    "movie_count": True,
                    "average_rating": ":.2f"
                }
            )

        elif chart_type == "bar":

            fig = px.bar(
                result,
                x=query_info["x"],
                y=query_info["y"],
                title=query_info["title"]
            )

        elif chart_type == "line":

            fig = px.line(
                result,
                x=query_info["x"],
                y=query_info["y"],
                markers=True,
                title=query_info["title"]
            )

        else:

            fig = px.scatter(
                result,
                x=query_info["x"],
                y=query_info["y"],
                hover_name=(
                    "title"
                    if "title" in result.columns
                    else None
                ),
                title=query_info["title"]
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "The query returned no records "
            "for the current filters."
        )

    # --------------------------------------------------------
    # Query insight
    # --------------------------------------------------------

    st.subheader("💡 Query Insight")

    st.info(query_info["insight"])


# ============================================================
# 15. MAIN APPLICATION
# ============================================================

def main():

    st.title("TMDB Movie Analytics Dashboard")

    data = load_all_csv_files()

    if data is None:
        st.error(
            "Unable to load CSV files from GitHub. "
            "Please check the repository URLs and file names."
        )
        st.stop()

    df = prepare_data(data)

    # Continue with your existing filters,
    # navigation, dashboard and query explorer code

        # Get genre names
        genres = sorted(
            data["genres"]["genre_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values != ""]
            .unique()
            .tolist()
        )

    except Exception as exc:

        st.error(f"Error loading TMDB data: {exc}")
        st.stop()

    if df.empty:

        st.error("No movie records found in movies.csv.")
        st.stop()

    # Initialize filter state
    genres, min_year, max_year, languages = seed_filter_state(
        df,
        genres
    )

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "Filters",
            "Dashboard",
            "Query Explorer"
        ]
    )

    # Render shared filters on every page
    render_sidebar_filters(
        genres,
        min_year,
        max_year,
        languages
    )

    # Apply filters across all pages
    filtered = apply_filters(df)

    st.sidebar.divider()

    st.sidebar.metric(
        "🎬 Matching Movies",
        f"{len(filtered):,}"
    )

    # Page selection
    if page == "Filters":

        render_filters(
            df,
            genres,
            min_year,
            max_year,
            languages
        )

    elif page == "Dashboard":

        render_dashboard(filtered)

    elif page == "Query Explorer":

        render_query_explorer(filtered)


# ============================================================
# 16. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()

