import streamlit as st
import pandas as pd
import altair as alt

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = r"prepared_batter_entry_dataset_with_expected_avg.csv"
df = pd.read_csv(DATA_PATH)

# Set page config
st.set_page_config(page_title="Test Batter Performance Judge", layout="wide")
st.title("Test Batter Performance Judge")

# =========================================================
#                GLOBAL SIDEBAR (common for both tabs)
# =========================================================
st.sidebar.header("Summary Filters")

# ---- 1. Year Range ----
year_min = int(df['year'].min())
year_max = int(df['year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    key="global_year_range"
)

# ---- 2. Batters ----
batters = sorted(df['bat'].unique())
batter_filter = st.sidebar.multiselect("Batters", batters, key="global_batters")

# ---- 3. Opponents ----
opponents = sorted(df['opponent'].unique())
opponent_filter = st.sidebar.multiselect("Opponents", opponents, key="global_opp")

# ---- 4. Countries ----
countries = sorted(df['country'].unique())
country_filter = st.sidebar.multiselect("Host Countries", countries, key="global_country")

# ---- 4b. Batting team (team_bat) ----
batting_teams = []
if 'team_bat' in df.columns:
    batting_teams = sorted(df['team_bat'].unique())
batting_team_filter = st.sidebar.multiselect("Batting Team", batting_teams, key="global_batting_team")

# ---- 5. Innings Number ----
innings_nums = sorted(df['inns_num'].unique())
inns_filter = st.sidebar.multiselect("Innings Number", innings_nums, key="global_inns")

# ---- 6. Position ---- (global)
position_map = {
    "Opener": 0,
    "Number 3": 1,
    "Number 4": 2,
    "Number 5": 3,
    "Number 6": 4,
    "Number 7": 5,
    "Number 8": 6,
    "Number 9": 7,
    "Number 10": 8,
    "Number 11": 9,
}
# allow selecting multiple positions for the summary filters (empty == all positions)
position_choices = st.sidebar.multiselect(
    "Batting Position (based on wickets fallen)",
    list(position_map.keys()),
    key="global_position"
)

# ---- 8. Conditional difficulty (summary) ----
# Labelled in the UI as expected_RPI but maps to the `expected_avg` column in the data
difficulty_options = [
    "hard (expected_RPI<30)",
    "moderate (30<=expected_RPI<50)",
    "easy (50<=expected_RPI)"
]
difficulty_choices = st.sidebar.multiselect(
    "Conditional difficulty",
    difficulty_options,
    key="global_difficulty"
)

# ---- 7. Ranking Filters Section ----
st.sidebar.markdown("---")
st.sidebar.subheader("Ranking Filters")

min_total_runs = st.sidebar.number_input(
    "Minimum Total Actual Runs",
    min_value=0,
    max_value=20000,
    value=5000,
    step=100,
    key="rank_minruns"
)

# Ranking filter selections
rank_mapping_country = st.sidebar.multiselect(
    "Host Country (Rankings)",
    sorted(df["country"].unique()),
    key="rank_country"
)

rank_mapping_opponent = st.sidebar.multiselect(
    "Opponent (Rankings)",
    sorted(df["opponent"].unique()),
    key="rank_opp"
)

# Ranking batting team filter
rank_mapping_batting_team = st.sidebar.multiselect(
    "Batting Team (Rankings)",
    sorted(df["team_bat"].unique()) if 'team_bat' in df.columns else [],
    key="rank_batting_team"
)

# Ranking position filter uses same mapping as global
rank_pos_strings = st.sidebar.multiselect(
    "Batting Position (Rankings)",
    list(position_map.keys()),
    key="rank_pos"
)
rank_mapping_pos = [position_map[p] for p in rank_pos_strings]

rank_mapping_inns = st.sidebar.multiselect(
    "Innings Number (Rankings)",
    sorted(df["inns_num"].unique()),
    key="rank_inns"
)

# ---- Ranking conditional difficulty ----
rank_mapping_difficulty = st.sidebar.multiselect(
    "Conditional difficulty (Rankings)",
    difficulty_options,
    key="rank_difficulty"
)

rank_year_range = st.sidebar.slider(
    "Rankings Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    key="rank_year_range"
)

rank_basis = st.sidebar.selectbox(
    "Rank By",
    [
        "Performance Factor (total actual runs / total expected runs)",
        "Consistency Factor (good innings / bad innings)"
    ],
    key="rank_basis"
)

# =========================================================
#                          TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Batter Summary", "Rankings", "Info"])

# =========================================================
#                         TAB 1 — SUMMARY
# =========================================================
with tab1:
    # Button to trigger summary generation
    if st.button("Generate Summary Results"):
        
        # Copy dataframe and apply global filters
        filtered = df.copy()
        filtered = filtered[
            (filtered['year'] >= year_range[0]) &
            (filtered['year'] <= year_range[1])
        ]
        if batter_filter:
            filtered = filtered[filtered['bat'].isin(batter_filter)]
        # apply batting team filter (if present in data)
        if batting_team_filter:
            if 'team_bat' in filtered.columns:
                filtered = filtered[filtered['team_bat'].isin(batting_team_filter)]
        if opponent_filter:
            filtered = filtered[filtered['opponent'].isin(opponent_filter)]
        if country_filter:
            filtered = filtered[filtered['country'].isin(country_filter)]
        if inns_filter:
            filtered = filtered[filtered['inns_num'].isin(inns_filter)]
        if position_choices:
            wicket_vals = [position_map[p] for p in position_choices]
            filtered = filtered[filtered['wickets_when_in'].isin(wicket_vals)]

        # ---- apply summary conditional difficulty filter (expected_RPI → expected_avg)
        if difficulty_choices:
            if 'expected_avg' in filtered.columns:
                mask = pd.Series(False, index=filtered.index)
                lower_choices = [c.lower() for c in difficulty_choices]
                if any('hard' in c for c in lower_choices):
                    mask |= (filtered['expected_avg'] < 30)
                if any('moderate' in c for c in lower_choices):
                    mask |= ((filtered['expected_avg'] >= 30) & (filtered['expected_avg'] < 50))
                if any('easy' in c for c in lower_choices):
                    mask |= (filtered['expected_avg'] >= 50)
                filtered = filtered[mask]
            else:
                st.warning("'expected_avg' column not found — conditional difficulty filtering (expected_RPI) will be skipped.")

        # Determine actual runs column dynamically
        if 'actual_runs' in filtered.columns:
            actual_col = 'actual_runs'
        elif 'y' in filtered.columns:
            actual_col = 'y'
        elif 'runs' in filtered.columns:
            actual_col = 'runs'
        else:
            possible = [c for c in filtered.columns if 'run' in c.lower()]
            actual_col = possible[0]

        if filtered.empty:
            st.warning("No data matches the filters.")
        else:
            # Compute averages and counts
            avg_actual = filtered[actual_col].mean()
            avg_pred = filtered['expected_avg'].mean()
            good_innings = (filtered[actual_col] > filtered['expected_avg']).sum()
            bad_innings = (filtered[actual_col] <= filtered['expected_avg']).sum()
            diff = avg_actual - avg_pred

            # Display KPIs
            st.subheader("Summary")
            c1, c2, c3, c4 = st.columns([2, 2, 1.5, 2])
            c1.metric("Actual runs / innings", f"{avg_actual:.2f}")
            c2.metric("Predicted runs / innings", f"{avg_pred:.2f}")
            c3.metric("Good innings", int(good_innings))
            c4.metric("Bad innings", int(bad_innings))

            # Performance message
            if diff > 0:
                st.success(f"Player is UP by +{diff:.2f} runs per innings")
            elif diff < 0:
                st.error(f"Player is DOWN by {abs(diff):.2f} runs per innings")
            else:
                st.info("Performance is exactly on expected baseline.")

            # ----------------------------
            # Year-by-year bar chart
            # ----------------------------
            st.markdown("---")
            chart_src = filtered[[actual_col, 'expected_avg', 'year']].copy()
            chart_src.rename(columns={actual_col: 'actual'}, inplace=True)
            year_agg = chart_src.groupby("year").mean().reset_index()

            # Melt for altair plotting
            melt_df = pd.melt(
                year_agg,
                id_vars=["year"],
                value_vars=["actual", "expected_avg"],
                var_name="type",
                value_name="avg_runs"
            ).replace({"expected_avg": "predicted"})

            # Define colors: actual = green, predicted = blue
            color_scale = alt.Scale(domain=["actual", "predicted"], range=["#2ecc71", "#3498db"])

            bar = alt.Chart(melt_df).mark_bar().encode(
                x='year:O',
                xOffset='type:N',
                y='avg_runs:Q',
                color=alt.Color('type:N', scale=color_scale),
                tooltip=['year', 'type', 'avg_runs']
            ).properties(title="Actual vs Predicted Runs per Innings (Yearly)")

            st.altair_chart(bar, use_container_width=True)

            # Expander to show filtered dataframe
            with st.expander("Show Filtered Data"):
                st.dataframe(filtered)

# =========================================================
#                         TAB 2 — RANKINGS
# =========================================================
with tab2:
    # Show a short tip about how to rank for a specific series
    st.info("Tip: To see matches between two countries, add both in the batting team and opponent filter. This helps generate rankings for a particular bilateral series.")

    # Button to trigger ranking generation
    if st.button("Generate Rankings"):
        # Copy dataframe and apply ranking filters
        rdf = df.copy()
        rdf = rdf[
            (rdf['year'] >= rank_year_range[0]) &
            (rdf['year'] <= rank_year_range[1])
        ]
        if rank_mapping_country:
            rdf = rdf[rdf["country"].isin(rank_mapping_country)]
        if rank_mapping_opponent:
            rdf = rdf[rdf["opponent"].isin(rank_mapping_opponent)]
        if rank_mapping_pos:
            rdf = rdf[rdf["wickets_when_in"].isin(rank_mapping_pos)]
        if rank_mapping_batting_team:
            if 'team_bat' in rdf.columns:
                rdf = rdf[rdf["team_bat"].isin(rank_mapping_batting_team)]
        if rank_mapping_inns:
            rdf = rdf[rdf["inns_num"].isin(rank_mapping_inns)]

        # ---- apply rankings conditional difficulty filter (expected_RPI -> expected_avg)
        if rank_mapping_difficulty:
            if 'expected_avg' in rdf.columns:
                lower_choices_r = [c.lower() for c in rank_mapping_difficulty]
                mask_r = pd.Series(False, index=rdf.index)
                if any('hard' in c for c in lower_choices_r):
                    mask_r |= (rdf['expected_avg'] < 30)
                if any('moderate' in c for c in lower_choices_r):
                    mask_r |= ((rdf['expected_avg'] >= 30) & (rdf['expected_avg'] < 50))
                if any('easy' in c for c in lower_choices_r):
                    mask_r |= (rdf['expected_avg'] >= 50)
                rdf = rdf[mask_r]
            else:
                st.warning("'expected_avg' column not found — conditional difficulty filtering (expected_RPI) will be skipped for rankings.")
        # Actual runs column
        if 'actual_runs' in rdf.columns:
            actual_col_r = 'actual_runs'
        elif 'y' in rdf.columns:
            actual_col_r = 'y'
        else:
            actual_col_r = [c for c in rdf.columns if 'run' in c.lower()][0]

        # Compute good and bad innings flags
        rdf["good_flag"] = (rdf[actual_col_r] > rdf["expected_avg"]).astype(int)
        rdf["bad_flag"] = (rdf[actual_col_r] <= rdf["expected_avg"]).astype(int)

        # Aggregate rankings
        rank_table = (
            rdf.groupby("bat")
            .agg(
                total_actual_runs=(actual_col_r, "sum"),
                total_expected_runs=("expected_avg", "sum"),
                good_innings=("good_flag", "sum"),
                bad_innings=("bad_flag", "sum")
            ).reset_index()
        )

        # Apply minimum total runs filter
        rank_table["total_expected_runs"] = rank_table["total_expected_runs"].round(0).astype(int)
        rank_table = rank_table[rank_table["total_actual_runs"] >= min_total_runs]

        # Compute ranking metrics
        rank_table["performance_factor"] = (
            rank_table["total_actual_runs"] /
            rank_table["total_expected_runs"]
        ).round(3)

        rank_table["consistency_factor"] = rank_table.apply(
            lambda r: round(r["good_innings"] / r["bad_innings"], 3)
            if r["bad_innings"] > 0 else float("inf"),
            axis=1
        )

        # Sort by chosen ranking basis
        if rank_basis.startswith("Performance"):
            rank_table = rank_table.sort_values("performance_factor", ascending=False)
        else:
            rank_table = rank_table.sort_values("consistency_factor", ascending=False)

        if rank_table.empty:
            st.warning("No player matches the ranking filters.")
        else:
            st.dataframe(rank_table.reset_index(drop=True))

with tab3:
    st.header("About the Model and Metrics")

    st.markdown("""
    ### Key Notes:
    - **Runs per innings ≠ batting average**. RPI does not treat outs and not outs differently.
    - **Good innings** = actual runs > predicted runs
    - **Bad innings** = actual runs ≤ predicted runs
    - **Data coverage**: Past 25 years of international Test cricket till the WTC 2025 Final.
        - **Performance Factor** = total actual runs / total predicted runs
        - **Consistency Factor** = good innings / bad innings
        - **expected_RPI** in the UI maps to the `expected_avg` column in the dataset —
            it's the model's expected runs per innings for the batter in that entry.
                
    ### Contextual Notes:
    - It’s impossible for a human mind to consider and evaluate all factors affecting a Test match knock simultaneously.
    - Even judging batters within the same generation is difficult, let alone across eras.
    - ML helps provide a more context-aware evaluation by comparing actual runs with the model’s expected runs.            

    ### Model Features:
    - Opponent
    - Host country
    - Score when the batter came in (runs and wickets)
    - Year
    - Innings number
    - Average runs of the previous innings of the match
    - Team's previous-innings runs
    - Opponent's previous-innings runs
    - Over number
    - Average innings score of the past year in the country

    The features are fed to a **CatBoostRegressor**, with runs scored by the batter as the target variable. This model is used primarily because it handles non-linear relationships well.

    
    """)


