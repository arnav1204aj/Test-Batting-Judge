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
# =========================================================
#                SIDEBAR MODE SELECTOR
# =========================================================
st.sidebar.header("Select View Mode")
mode = st.sidebar.radio(
    "Choose what you want to analyse:",
    ["Summary", "Rankings", "Comparison"],
    index=0
)


# =========================================================
#                COMMON FILTER STRUCTURE
# =========================================================
year_min = int(df['year'].min())
year_max = int(df['year'].max())
batters = sorted(df['bat'].unique())
opponents = sorted(df['opponent'].unique())
countries = sorted(df['country'].unique())
innings_nums = sorted(df['inns_num'].unique())

position_map = {
    "Opener": 0,  "Number 3": 1, "Number 4": 2, "Number 5": 3,
    "Number 6": 4, "Number 7": 5, "Number 8": 6, "Number 9": 7,
    "Number 10": 8, "Number 11": 9,
}

difficulty_options = [
    "hard (expected_RPI<30)",
    "moderate (30<=expected_RPI<50)",
    "easy (50<=expected_RPI)",
]

# =========================================================
#       FILTERS FOR SUMMARY + COMPARISON MODES
# =========================================================
def summary_and_comparison_filters(prefix=""):
    """Reusable filter blocks for Summary and Comparison tabs."""
    yr = st.sidebar.slider(
        f"{prefix}Year Range",
        min_value=year_min, max_value=year_max,
        value=(year_min, year_max),
        key=f"{prefix}_year_range"
    )
    bt = st.sidebar.multiselect(f"{prefix}Batters", batters, key=f"{prefix}_batters")
    opp = st.sidebar.multiselect(f"{prefix}Opponents", opponents, key=f"{prefix}_opp")
    ct = st.sidebar.multiselect(f"{prefix}Host Countries", countries, key=f"{prefix}_ct")
    inns = st.sidebar.multiselect(f"{prefix}Innings Number", innings_nums, key=f"{prefix}_inns")
    pos = st.sidebar.multiselect(f"{prefix}Batting Position", list(position_map.keys()), key=f"{prefix}_pos")
    cond = st.sidebar.multiselect(f"{prefix}Condition Difficulty", difficulty_options, key=f"{prefix}_diff")

    team_bat = sorted(df["team_bat"].unique()) if "team_bat" in df.columns else []
    tbat = st.sidebar.multiselect(f"{prefix}Batting Team", team_bat, key=f"{prefix}_team_bat")

    return {
        "year_range": yr,
        "batters": bt,
        "opponent": opp,
        "country": ct,
        "inns": inns,
        "pos": pos,
        "diff": cond,
        "team_bat": tbat,
    }

# =========================================================
#         FILTERS FOR COMPARISON MODE (new)
# =========================================================
def comparison_filters(prefix="cmp"):
    st.sidebar.subheader("Comparison Filters")

    # Multiple batters
    batter_list = st.sidebar.multiselect("Select Batters to Compare", batters, key=f"{prefix}_batters")

    yr = st.sidebar.slider(
        "Comparison Year Range", year_min, year_max,
        (year_min, year_max), key=f"{prefix}_year"
    )

    opp = st.sidebar.multiselect("Opponents", opponents, key=f"{prefix}_opp")
    ct = st.sidebar.multiselect("Host Countries", countries, key=f"{prefix}_ct")
    inns = st.sidebar.multiselect("Innings Number", innings_nums, key=f"{prefix}_inns")
    pos = st.sidebar.multiselect("Batting Position", list(position_map.keys()), key=f"{prefix}_pos")
    cond = st.sidebar.multiselect("Condition Difficulty", difficulty_options, key=f"{prefix}_diff")

    # IMPORTANT → No batting team filter for comparison

    return {
        "batters": batter_list,
        "year_range": yr,
        "opp": opp,
        "ct": ct,
        "inns": inns,
        "pos": pos,
        "diff": cond,
    }

# =========================================================
#        FILTERS FOR RANKINGS MODE (unchanged)
# =========================================================
def rankings_filters():
    st.sidebar.subheader("Ranking Filters")
    yr = st.sidebar.slider("Rankings Year Range", year_min, year_max, (year_min, year_max))
    min_runs = st.sidebar.number_input("Minimum Total Actual Runs", min_value=0, value=5000)
    ct = st.sidebar.multiselect("Host Country", countries)
    opp = st.sidebar.multiselect("Opponent", opponents)
    tbat = st.sidebar.multiselect("Batting Team", sorted(df["team_bat"].unique()))
    pos = st.sidebar.multiselect("Batting Position", list(position_map.keys()))
    inns = st.sidebar.multiselect("Innings Number", innings_nums)
    diff = st.sidebar.multiselect("Condition Difficulty", difficulty_options)

    basis = st.sidebar.selectbox(
        "Rank By",
        ["Performance Factor (total actual runs / total expected runs)",
         "Consistency Factor (good innings / bad innings)"]
    )

    return {
        "year_range": yr,
        "min_runs": min_runs,
        "ct": ct, "opp": opp, "tbat": tbat,
        "pos": pos, "inns": inns, "diff": diff,
        "basis": basis,
    }

# ---- Load Sidebar Filters Depending on Mode ----

filters = None

if mode == "Summary":
    filters = summary_and_comparison_filters()

elif mode == "Rankings":
    filters = rankings_filters()

elif mode == "Comparison":
    filters = comparison_filters()

# ---- Unpack filters depending on mode ----
if mode == "Summary" and filters:
    year_range = filters["year_range"]
    batter_filter = filters["batters"]
    opponent_filter = filters["opponent"]
    country_filter = filters["country"]
    inns_filter = filters["inns"]
    position_choices = filters["pos"]
    difficulty_choices = filters["diff"]
    batting_team_filter = filters["team_bat"]

elif mode == "Rankings" and filters:
    rank_year_range = filters["year_range"]
    min_total_runs = filters["min_runs"]
    rank_mapping_country = filters["ct"]
    rank_mapping_opponent = filters["opp"]
    rank_mapping_batting_team = filters["tbat"]
    rank_mapping_pos = filters["pos"]
    rank_mapping_inns = filters["inns"]
    rank_mapping_difficulty = filters["diff"]
    rank_basis = filters["basis"]

elif mode == "Comparison" and filters:
    cmp_batters = filters["batters"]
    cmp_year_range = filters["year_range"]
    cmp_opp = filters["opp"]
    cmp_ct = filters["ct"]
    cmp_inns = filters["inns"]
    cmp_pos = filters["pos"]
    cmp_diff = filters["diff"]

# ----------------------------
# Ensure defaults for any missing filter variables
# This prevents NameError when users navigate without selecting a mode
# ----------------------------
if 'year_range' not in globals():
    year_range = (year_min, year_max)
if 'batter_filter' not in globals():
    batter_filter = []
if 'opponent_filter' not in globals():
    opponent_filter = []
if 'country_filter' not in globals():
    country_filter = []
if 'inns_filter' not in globals():
    inns_filter = []
if 'position_choices' not in globals():
    position_choices = []
if 'difficulty_choices' not in globals():
    difficulty_choices = []
if 'batting_team_filter' not in globals():
    batting_team_filter = []

if 'rank_year_range' not in globals():
    rank_year_range = (year_min, year_max)
if 'min_total_runs' not in globals():
    min_total_runs = 5000
if 'rank_mapping_country' not in globals():
    rank_mapping_country = []
if 'rank_mapping_opponent' not in globals():
    rank_mapping_opponent = []
if 'rank_mapping_batting_team' not in globals():
    rank_mapping_batting_team = []
if 'rank_mapping_pos' not in globals():
    rank_mapping_pos = []
if 'rank_mapping_inns' not in globals():
    rank_mapping_inns = []
if 'rank_mapping_difficulty' not in globals():
    rank_mapping_difficulty = []
if 'rank_basis' not in globals():
    rank_basis = "Performance Factor (total actual runs / total expected runs)"

if 'cmp_batters' not in globals():
    cmp_batters = []
if 'cmp_year_range' not in globals():
    cmp_year_range = (year_min, year_max)
if 'cmp_opp' not in globals():
    cmp_opp = []
if 'cmp_ct' not in globals():
    cmp_ct = []
if 'cmp_inns' not in globals():
    cmp_inns = []
if 'cmp_pos' not in globals():
    cmp_pos = []
if 'cmp_diff' not in globals():
    cmp_diff = []

# =========================================================
#                      TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Batter Summary", "Rankings", "Comparison", "Info"])


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
            st.markdown("### Actual vs Predicted Runs per Innings (Yearly)")
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

            

            # =========================================================
#   PERFORMANCE FACTOR PDF (per-innings distribution)
# =========================================================

            st.markdown("### Innings Performance Factor Distribution")
            st.caption("Performance Factor = actual runs / predicted runs")

            # Compute per-innings PF
            pf_df = filtered.copy()
            pf_df["pf"] = pf_df[actual_col] / pf_df["expected_avg"]

            # Kernel Density Estimation for smooth PDF
            pdf_chart = (
                alt.Chart(pf_df)
                .transform_density(
                    "pf",
                    as_=["pf", "density"],
                    extent=[0, max(3, pf_df["pf"].max())],   # ensure full range
                    steps=200
                )
                .mark_area(opacity=0.45)
                .encode(
                    x=alt.X("pf:Q", title="Performance Factor (actual / predicted)"),
                    y=alt.Y("density:Q", title="Probability Density"),
                    tooltip=["pf:Q", "density:Q"]
                )
                .properties(height=300)
            )

            # Vertical markers
            vlines = alt.Chart(pd.DataFrame({
                "x": [0, 0.5, 1, 2, 3],
                "label": ["duck", "poor", "average", "good", "exceptional"]
            })).mark_rule(color="red").encode(
                x="x:Q",
                tooltip=["label", "x:Q"]
            )

            text_labels = alt.Chart(pd.DataFrame({
                "x": [0, 0.5, 1, 2, 3],
                "y": [0]*5,
                "label": ["duck", "poor", "average", "good", "exceptional"]
            })).mark_text(
                dy=-5, color="red", fontSize=12
            ).encode(
                x="x:Q",
                y=alt.value(0),
                text="label:N"
            )

            # Combine PDF + vertical markers
            st.altair_chart(pdf_chart + vlines + text_labels, use_container_width=True)
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

# =========================================================
#                         TAB 3 — COMPARISON
# =========================================================
with tab3:
    st.header("Compare Multiple Batters")

    if st.button("Generate Comparison"):
        if not cmp_batters:
            st.warning("Please select at least one batter to compare.")
        else:
            st.subheader(f"Comparing: {', '.join(cmp_batters)}")
            
            # Apply filters function
            def apply_filters(sub_df, name):
                df2 = sub_df.copy()
                df2 = df2[(df2["bat"] == name)]
                df2 = df2[
                    (df2["year"] >= cmp_year_range[0]) &
                    (df2["year"] <= cmp_year_range[1])
                ]

                if cmp_opp:
                    df2 = df2[df2["opponent"].isin(cmp_opp)]
                if cmp_ct:
                    df2 = df2[df2["country"].isin(cmp_ct)]
                if cmp_inns:
                    df2 = df2[df2["inns_num"].isin(cmp_inns)]
                if cmp_pos:
                    wicket_vals = [position_map[p] for p in cmp_pos]
                    df2 = df2[df2["wickets_when_in"].isin(wicket_vals)]

                # difficulty filter (expected_avg)
                if cmp_diff and "expected_avg" in df2.columns:
                    mask = pd.Series(False, index=df2.index)
                    lower = [c.lower() for c in cmp_diff]
                    if any("hard" in c for c in lower):
                        mask |= (df2["expected_avg"] < 30)
                    if any("moderate" in c for c in lower):
                        mask |= ((df2["expected_avg"] >= 30) & (df2["expected_avg"] < 50))
                    if any("easy" in c for c in lower):
                        mask |= (df2["expected_avg"] >= 50)
                    df2 = df2[mask]

                return df2

            # Apply filters to all selected batters
            filtered_dfs = {batter: apply_filters(df, batter) for batter in cmp_batters}
            
            # Check if any batter has data
            if all(v.empty for v in filtered_dfs.values()):
                st.warning("No data found for selected batters with these filters.")
            else:
                # Determine actual runs column
                def get_actual(df_x):
                    if "actual_runs" in df_x.columns: return "actual_runs"
                    if "y" in df_x.columns: return "y"
                    return [c for c in df_x.columns if "run" in c.lower()][0]

                act_col = get_actual(df)

                # Compute metrics for all batters
                def overall_metrics(df_x, act_col):
                    if df_x.empty:
                        return {
                            "performance_factor": float("nan"),
                            "consistency_factor": float("nan")
                        }
                    total_actual = df_x[act_col].sum()
                    total_expected = df_x["expected_avg"].sum() if "expected_avg" in df_x.columns else 0
                    pf = (total_actual / total_expected) if total_expected else float("nan")
                    good = int((df_x[act_col] > df_x["expected_avg"]).sum()) if "expected_avg" in df_x.columns else 0
                    bad = int((df_x[act_col] <= df_x["expected_avg"]).sum()) if "expected_avg" in df_x.columns else 0
                    cons = (good / bad) if bad > 0 else float("inf")
                    return {
                        "performance_factor": round(pf, 3) if not pd.isna(pf) else float("nan"),
                        "consistency_factor": round(cons, 3) if cons != float("inf") else float("inf")
                    }

                # Build comparison table
                metrics_list = []
                for batter in cmp_batters:
                    metrics = overall_metrics(filtered_dfs[batter], act_col)
                    metrics_list.append({
                        "Batter": batter,
                        "Performance Factor": metrics["performance_factor"],
                        "Consistency Factor": metrics["consistency_factor"]
                    })
                
                metrics_df = pd.DataFrame(metrics_list)
                st.subheader("Comparison Table")
                st.dataframe(metrics_df, use_container_width=True)

                # Yearly Performance Factor line chart
                def per_innings_pf(df_x, act_col):
                    if df_x.empty:
                        return pd.DataFrame()
                    d = df_x[df_x['expected_avg'] > 0].copy()
                    d = d.assign(pf=(d[act_col] / d['expected_avg']))
                    return d

                year_dfs = []
                for batter in cmp_batters:
                    pf_data = per_innings_pf(filtered_dfs[batter], act_col)
                    if not pf_data.empty:
                        year_agg = pf_data.groupby('year')['pf'].mean().reset_index().assign(bat=batter)
                        year_dfs.append(year_agg)
                
                if year_dfs:
                    year_df = pd.concat(year_dfs, ignore_index=True)
                    
                    year_chart = alt.Chart(year_df).mark_line(point=True).encode(
                        x=alt.X('year:O', title='Year'),
                        y=alt.Y('pf:Q', title='Mean Performance Factor'),
                        color=alt.Color('bat:N'),
                        tooltip=[alt.Tooltip('bat:N'), alt.Tooltip('year:O'), alt.Tooltip('pf:Q', format='.3f')]
                    ).properties(title='Yearly Performance Factor — All Batters')

                    st.altair_chart(year_chart, use_container_width=True)

                    # PDF (density) of per-innings Performance Factor — one plot per batter
                    pf_dfs = []
                    for batter in cmp_batters:
                        pf_data = per_innings_pf(filtered_dfs[batter], act_col)
                        if not pf_data.empty:
                            pf_dfs.append(pf_data.assign(bat=batter))
                    
                    if pf_dfs:
                        pf_comb = pd.concat(pf_dfs, ignore_index=True)
                        pf_min = pf_comb['pf'].min()
                        pf_max = max(3, pf_comb['pf'].max())
                        
                        # Create separate density plot for each batter (stacked vertically)
                        for idx, batter in enumerate(cmp_batters):
                            batter_data = pf_comb[pf_comb['bat'] == batter]
                            if not batter_data.empty and len(batter_data) > 1:
                                # Use Altair's tableau10 color scheme (same as yearly PF)
                                color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                                batter_color = color_palette[idx % len(color_palette)]
                                
                                density = alt.Chart(batter_data).transform_density(
                                    'pf',
                                    as_=['pf', 'density'],
                                    extent=[pf_min, pf_max],
                                    steps=200
                                ).mark_area(opacity=0.6, color=batter_color).encode(
                                    x=alt.X('pf:Q', title='Performance Factor (actual / predicted)'),
                                    y=alt.Y('density:Q', title='Density'),
                                    tooltip=['pf:Q', 'density:Q']
                                ).properties(
                                    title=f'PDF — {batter}',
                                    height=200
                                )
                                
                                st.altair_chart(density, use_container_width=True)

                # Show raw data if requested
                with st.expander("Show DataFrames Used"):
                    for batter in cmp_batters:
                        if not filtered_dfs[batter].empty:
                            st.write(f"Data for {batter}")
                            st.dataframe(filtered_dfs[batter])

with tab4:
    st.header("About the Model and Metrics")

    st.markdown("""
    ### Key Notes:
    - **Runs per innings ≠ batting average**. RPI does not treat outs and not outs differently.
    - **Good innings** = actual runs > predicted runs
    - **Bad innings** = actual runs ≤ predicted runs
    - **Data coverage**: Past 25 years of international Test cricket (2001-2025).
        - **Performance Factor** = total actual runs / total predicted runs
        - **Consistency Factor** = good innings / bad innings
        - **expected_RPI** in the UI maps to the `expected_avg` column in the dataset —
            it's the model's expected runs per innings for the batter in that entry.
    
    - Read more: [Context is King — Substack](https://arnavj.substack.com/p/context-is-king)
                
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
