import streamlit as st
import pandas as pd

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = r"prepared_batter_entry_dataset_with_expected_avg.csv"
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Test Batter Performance Judge", layout="wide")

st.title("Test Batter Performance Judge")

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------

st.sidebar.header("Filters")

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
