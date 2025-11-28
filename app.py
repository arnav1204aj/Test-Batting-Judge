import streamlit as st
import pandas as pd

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = r"prepared_batter_entry_dataset_with_expected_avg.csv"
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Test Batter Performance Judge", layout="wide")

st.title("Test Batter Performance Judge")
# =========================================================
#   TABS: MAIN DASHBOARD  |  RANKINGS
# =========================================================
tab1, tab2 = st.tabs(["Batter Summary", "Rankings"])

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
with tab1:
    # your full existing code here

    st.sidebar.header("Summary Filters")

    # 1. Year filter (range slider)
    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    # 2. Batter filter
    batters = sorted(df['bat'].unique())
    batter_filter = st.sidebar.multiselect("Select Batters", batters)

    # 3. Opponent filter
    opponents = sorted(df['opponent'].unique())
    opponent_filter = st.sidebar.multiselect("Select Opponents", opponents)

    # 4. Country filter
    countries = sorted(df['country'].unique())
    country_filter = st.sidebar.multiselect("Select Host Countries", countries)

    # 5. Inns_num filter
    innings_nums = sorted(df['inns_num'].unique())
    inns_filter = st.sidebar.multiselect("Select Inns Number", innings_nums)

    # 6. Position filter
    #    0 → Opener, 1 → No.3, etc.
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

    position_choice = st.sidebar.selectbox(
        "Select Batting Position (based on wickets fallen)",
        ["All Positions"] + list(position_map.keys())
    )

    # ----------------------------
    # APPLY FILTERS
    # ----------------------------

    filtered = df.copy()

    filtered = filtered[
        (filtered['year'] >= year_range[0]) &
        (filtered['year'] <= year_range[1])
    ]

    if batter_filter:
        filtered = filtered[filtered['bat'].isin(batter_filter)]

    if opponent_filter:
        filtered = filtered[filtered['opponent'].isin(opponent_filter)]

    if country_filter:
        filtered = filtered[filtered['country'].isin(country_filter)]

    if inns_filter:
        filtered = filtered[filtered['inns_num'].isin(inns_filter)]

    if position_choice != "All Positions":
        wicket_val = position_map[position_choice]
        filtered = filtered[filtered['wickets_when_in'] == wicket_val]

    # ----------------------------
    # CALCULATIONS + UI
    # ----------------------------

    if filtered.empty:
        st.warning("No data matches these filters.")
    else:
        # robust lookup for actual runs column (some datasets use different names)
        if 'actual_runs' in filtered.columns:
            actual_col = 'actual_runs'
        elif 'y' in filtered.columns:
            actual_col = 'y'
        elif 'y_runs' in filtered.columns:
            actual_col = 'y_runs'
        else:
            # fallback to trying known alternatives
            possible = [c for c in filtered.columns if 'run' in c.lower()]
            actual_col = possible[0] if possible else None

        if actual_col is None:
            st.error("Couldn't find a column containing actual runs (looked for 'actual_runs', 'y', 'y_runs'). Please inspect the data.")
        else:
            avg_actual = filtered[actual_col].mean()
            avg_pred = filtered['expected_avg'].mean()
            good_innings = (filtered[actual_col] > filtered['expected_avg']).sum()
            bad_innings = (filtered[actual_col] <= filtered['expected_avg']).sum()

            # Difference per innings
            diff = avg_actual - avg_pred

            # Top line — summary KPIs in a single row
            st.subheader("Summary")
            k1, k2, k3, k4 = st.columns([2, 2, 1.5, 2])
            k1.metric("Actual runs / innings", f"{avg_actual:.2f}")
            k2.metric("Predicted runs / innings", f"{avg_pred:.2f}")
            k3.metric("Good innings", int(good_innings))
            k4.metric("Bad innings", int(bad_innings))

            # Up/Down indicator: color-coded and sports-y language
            if diff > 0:
                st.success(f"Player is UP by +{diff:.2f} runs per innings (beat expected)")
            elif diff < 0:
                st.error(f"Player is DOWN by {abs(diff):.2f} runs per innings (below expected)")
            else:
                st.info("Player performance is on-target vs expectations (0.00 runs difference) ")

            # Simple year-by-year grouped bars (Actual vs Predicted) — more space for the chart,
            # and remove the distribution plot to keep the dashboard focused for sports analytics.
            st.markdown("---")

            # Plot: year grouped comparison (side-by-side bars per year)
            import altair as alt

            chart_src = filtered[[actual_col, 'expected_avg', 'year']].copy()
            chart_src = chart_src.rename(columns={actual_col: 'actual'})
            
            # Aggregate by year
            year_agg = chart_src.groupby('year').agg({'actual':'mean','expected_avg':'mean'}).reset_index()
            year_df = pd.melt(year_agg, id_vars=['year'], value_vars=['actual','expected_avg'], var_name='type', value_name='avg_runs')
            # normalize label for the legend: show expected_avg as 'predicted'
            year_df['type'] = year_df['type'].replace({'expected_avg': 'predicted'})

            # grouped bars: use xOffset so actual and expected appear side-by-side for each year
            year_bar = alt.Chart(year_df).mark_bar().encode(
                x=alt.X('year:O', title='Year'),
                xOffset='type:N',
                y=alt.Y('avg_runs:Q', title='Avg runs per innings'),
                color=alt.Color('type:N', title='Series', scale=alt.Scale(domain=['actual','predicted'], range=['#2ca02c','#1f77b4'])),
                tooltip=[alt.Tooltip('year:O'), alt.Tooltip('type:N'), alt.Tooltip('avg_runs:Q', format='.2f')]
            ).properties(title='Actual vs Predicted runs per innings — year by year')

            # Use full width for the bar chart
            st.altair_chart(year_bar, use_container_width=True)

            # NOTE: Runs per innings is NOT the same as batting average — average differentiates outs vs not-outs.
            st.caption("Note: 'runs per innings' here counts runs per entry (all innings), and is NOT the same as batting average — batting average divides total runs by dismissals and treats not-outs differently. Good Innings are where actual runs are better than predicted. " \
            "Data till WTC 2025 Final.")

            st.markdown("---")

            # Optional: show filtered data
            with st.expander("Show Filtered Data"):
                st.dataframe(filtered)

# =========================================================
#                     RANKINGS TAB
# =========================================================
with tab2:

    st.header("Batter Rankings")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ranking Filters")

    # -------------------------------
    # 1. MIN TOTAL RUNS
    # -------------------------------
    min_total_runs = st.sidebar.number_input(
        "Minimum Total Actual Runs",
        min_value=0,
        max_value=20000,
        value=5000,
        step=100,
        key="rank_min_runs"
    )

    # -------------------------------
    # 2. Host Country Filter
    # -------------------------------
    rank_country = st.sidebar.multiselect(
        "Host Country",
        sorted(df["country"].unique()),
        key="rank_country"
    )

    # -------------------------------
    # 3. Opponent Filter
    # -------------------------------
    rank_opponent = st.sidebar.multiselect(
        "Opponent",
        sorted(df["opponent"].unique()),
        key="rank_opponent"
    )

    # -------------------------------
    # 4. Batting Position
    # -------------------------------
    rank_pos = st.sidebar.multiselect(
        "Batting Position",
        sorted(df["wickets_when_in"].unique()),
        key="rank_pos"
    )

    # -------------------------------
    # 5. Innings Number
    # -------------------------------
    rank_inns = st.sidebar.multiselect(
        "Innings Number",
        sorted(df["inns_num"].unique()),
        key="rank_inns"
    )

    # -------------------------------
    # 6. Year Range
    # -------------------------------
    r_year_min = int(df["year"].min())
    r_year_max = int(df["year"].max())

    rank_year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=r_year_min,
        max_value=r_year_max,
        value=(r_year_min, r_year_max),
        key="rank_year_slider"
    )

    # -------------------------------
    # 7. Ranking Basis
    # -------------------------------
    rank_basis = st.sidebar.selectbox(
        "Rank By",
        [
            "Performance Factor (total actual runs / total expected runs)",
            "Consistency Factor (good innings / bad innings)"
        ],
        key="rank_basis"
    )

    # -----------------------------------------------------
    # Definitions
    # -----------------------------------------------------
    st.info(
        "**Performance Factor** = Total Actual Runs ÷ Total Expected Runs\n\n"
        "**Consistency Factor** = Good Innings ÷ Bad Innings\n\n"
        "*Good Innings*: actual > expected\n\n"
        "*Bad Innings*: actual ≤ expected"
    )

    # =====================================================
    # APPLY FILTERS
    # =====================================================
    rdf = df.copy()

    rdf = rdf[
        (rdf["year"] >= rank_year_range[0]) &
        (rdf["year"] <= rank_year_range[1])
    ]

    if rank_country:
        rdf = rdf[rdf["country"].isin(rank_country)]

    if rank_opponent:
        rdf = rdf[rdf["opponent"].isin(rank_opponent)]

    if rank_pos:
        rdf = rdf[rdf["wickets_when_in"].isin(rank_pos)]

    if rank_inns:
        rdf = rdf[rdf["inns_num"].isin(rank_inns)]

    # -------------------------
        # Identify actual runs column
        # -------------------------
        # -------------------------
    # Identify actual runs column
    # -------------------------
    if 'actual_runs' in rdf.columns:
        actual_col = 'actual_runs'
    elif 'y' in rdf.columns:
        actual_col = 'y'
    else:
        possible = [c for c in rdf.columns if 'run' in c.lower()]
        actual_col = possible[0]

    # -------------------------
    # Create good/bad innings flags (row-wise)
    # -------------------------
    rdf["good_flag"] = (rdf[actual_col] > rdf["expected_avg"]).astype(int)
    rdf["bad_flag"]  = (rdf[actual_col] <= rdf["expected_avg"]).astype(int)

    # -------------------------
    # Group (VALID AGG SYNTAX)
    # -------------------------
    rank_table = (
        rdf.groupby("bat")
        .agg(
            total_actual_runs=(actual_col, "sum"),
            total_expected_runs=("expected_avg", "sum"),
            good_innings=("good_flag", "sum"),
            bad_innings=("bad_flag", "sum")
        )
        .reset_index()
    )

    rank_table["total_expected_runs"] = rank_table["total_expected_runs"].round(0).astype(int)
    # Filter by runs
    rank_table = rank_table[rank_table["total_actual_runs"] >= min_total_runs]

    # -------------------------
    # Ranking factors
    # -------------------------
    rank_table["performance_factor"] = (
        rank_table["total_actual_runs"] /
        rank_table["total_expected_runs"]
    ).round(3)

    rank_table["consistency_factor"] = rank_table.apply(
        lambda r: round(r["good_innings"] / r["bad_innings"], 3)
        if r["bad_innings"] > 0 else float("inf"),
        axis=1
    )

    # -------------------------
    # Sorting
    # -------------------------
    if rank_basis.startswith("Performance"):
        rank_table = rank_table.sort_values(
            "performance_factor", ascending=False
        )
    else:
        rank_table = rank_table.sort_values(
            "consistency_factor", ascending=False
        )

    # -------------------------
    # Display
    # -------------------------
    st.subheader("Rankings Table")

    if rank_table.empty:
        st.warning("No player matches the filters.")
    else:
        st.dataframe(rank_table.reset_index(drop=True))

