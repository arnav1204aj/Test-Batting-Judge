import streamlit as st
import pandas as pd
import altair as alt

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = r"prepared_batter_entry_dataset_with_expected_avg.csv"
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Test Batter Performance Judge", layout="wide")
st.title("Test Batter Performance Judge")

# =========================================================
#                GLOBAL SIDEBAR (common for both tabs)
# =========================================================
st.sidebar.header("Global Filters")

# ---- 1. Year Range ----
year_min = int(df['year'].min())
year_max = int(df['year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# ---- 2. Batters ----
batters = sorted(df['bat'].unique())
batter_filter = st.sidebar.multiselect("Batters", batters)

# ---- 3. Opponents ----
opponents = sorted(df['opponent'].unique())
opponent_filter = st.sidebar.multiselect("Opponents", opponents)

# ---- 4. Countries ----
countries = sorted(df['country'].unique())
country_filter = st.sidebar.multiselect("Host Countries", countries)

# ---- 5. Innings Number ----
innings_nums = sorted(df['inns_num'].unique())
inns_filter = st.sidebar.multiselect("Innings Number", innings_nums)

# ---- 6. Position ----
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
    "Batting Position (based on wickets fallen)",
    ["All Positions"] + list(position_map.keys())
)

# ---- 7. Ranking Filters Section ----
st.sidebar.markdown("---")
st.sidebar.subheader("Ranking Filters")

min_total_runs = st.sidebar.number_input(
    "Minimum Total Actual Runs",
    min_value=0,
    max_value=20000,
    value=5000,
    step=100
)

rank_mapping_country = st.sidebar.multiselect(
    "Host Country (Rankings)",
    sorted(df["country"].unique()),
)

rank_mapping_opponent = st.sidebar.multiselect(
    "Opponent (Rankings)",
    sorted(df["opponent"].unique()),
)

rank_mapping_pos = st.sidebar.multiselect(
    "Batting Position (Rankings)",
    sorted(df["wickets_when_in"].unique()),
)

rank_mapping_inns = st.sidebar.multiselect(
    "Innings Number (Rankings)",
    sorted(df["inns_num"].unique()),
)

rank_year_range = st.sidebar.slider(
    "Rankings Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

rank_basis = st.sidebar.selectbox(
    "Rank By",
    [
        "Performance Factor (total actual runs / total expected runs)",
        "Consistency Factor (good innings / bad innings)"
    ]
)

# =========================================================
#                       APPLY GLOBAL FILTERS
# =========================================================
filtered = df.copy()

# Apply filters
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

# Identify actual runs column
if 'actual_runs' in filtered.columns:
    actual_col = 'actual_runs'
elif 'y' in filtered.columns:
    actual_col = 'y'
elif 'runs' in filtered.columns:
    actual_col = 'runs'
else:
    possible = [c for c in filtered.columns if 'run' in c.lower()]
    actual_col = possible[0] if possible else None


# =========================================================
#                        TABS
# =========================================================
tab1, tab2 = st.tabs(["Batter Summary", "Rankings"])

# =========================================================
#                       TAB 1 — SUMMARY
# =========================================================
with tab1:

    if filtered.empty:
        st.warning("No data matches the filters.")
    else:

        avg_actual = filtered[actual_col].mean()
        avg_pred = filtered['expected_avg'].mean()

        good_innings = (filtered[actual_col] > filtered['expected_avg']).sum()
        bad_innings = (filtered[actual_col] <= filtered['expected_avg']).sum()

        diff = avg_actual - avg_pred

        # KPIs
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
        # Year-by-year chart
        # ----------------------------
        st.markdown("---")
        chart_src = filtered[[actual_col, 'expected_avg', 'year']].copy()
        chart_src.rename(columns={actual_col: 'actual'}, inplace=True)

        year_agg = chart_src.groupby("year").mean().reset_index()
        melt_df = pd.melt(
            year_agg,
            id_vars=["year"],
            value_vars=["actual", "expected_avg"],
            var_name="type",
            value_name="avg_runs"
        )
        melt_df['type'] = melt_df['type'].replace({"expected_avg": "predicted"})

        bar = alt.Chart(melt_df).mark_bar().encode(
            x='year:O',
            xOffset='type:N',
            y='avg_runs:Q',
            color='type:N',
            tooltip=['year', 'type', 'avg_runs']
        ).properties(title="Actual vs Predicted Runs per Innings (Yearly)")

        st.altair_chart(bar, use_container_width=True)

        st.caption(
            "Note: Runs per innings ≠ batting average. "
            "Good innings = actual > predicted."
        )

        with st.expander("Show Filtered Data"):
            st.dataframe(filtered)


# =========================================================
#                   TAB 2 — RANKINGS
# =========================================================
with tab2:

    st.header("Batter Rankings")

    # Apply ranking filters to new df
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
    if rank_mapping_inns:
        rdf = rdf[rdf["inns_num"].isin(rank_mapping_inns)]

    # Actual runs column
    if 'actual_runs' in rdf.columns:
        actual_col_r = 'actual_runs'
    elif 'y' in rdf.columns:
        actual_col_r = 'y'
    else:
        actual_col_r = [c for c in rdf.columns if 'run' in c.lower()][0]

    rdf["good_flag"] = (rdf[actual_col_r] > rdf["expected_avg"]).astype(int)
    rdf["bad_flag"] = (rdf[actual_col_r] <= rdf["expected_avg"]).astype(int)

    rank_table = (
        rdf.groupby("bat")
        .agg(
            total_actual_runs=(actual_col_r, "sum"),
            total_expected_runs=("expected_avg", "sum"),
            good_innings=("good_flag", "sum"),
            bad_innings=("bad_flag", "sum")
        ).reset_index()
    )

    rank_table["total_expected_runs"] = rank_table["total_expected_runs"].round(0).astype(int)

    # Minimum runs filter
    rank_table = rank_table[rank_table["total_actual_runs"] >= min_total_runs]

    # Ranking metrics
    rank_table["performance_factor"] = (
        rank_table["total_actual_runs"] /
        rank_table["total_expected_runs"]
    ).round(3)

    rank_table["consistency_factor"] = rank_table.apply(
        lambda r: round(r["good_innings"] / r["bad_innings"], 3)
        if r["bad_innings"] > 0 else float("inf"),
        axis=1
    )

    if rank_basis.startswith("Performance"):
        rank_table = rank_table.sort_values("performance_factor", ascending=False)
    else:
        rank_table = rank_table.sort_values("consistency_factor", ascending=False)

    if rank_table.empty:
        st.warning("No player matches the ranking filters.")
    else:
        st.dataframe(rank_table.reset_index(drop=True))

