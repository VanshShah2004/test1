#!/usr/bin/env python3
"""
Interactive Resume Scoring Dashboard (Streamlit)

Run: streamlit run dashboard.py

This app reads pipeline outputs from outputs/scoring_results.json and renders
interactive charts for per-criterion comparisons, total score distribution,
and candidate vs job benchmarks. It supports selection of one or more
candidates, click/hover details, and auto-updates if new resumes are added
and the pipeline re-saves JSON.
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


OUTPUT_JSON = os.path.join("outputs", "scoring_results.json")


@st.cache_data(show_spinner=False)
def load_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_file_last_updated(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown"


def results_to_long_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for r in results:
        if not r.get("success"):
            continue
        s = r["scoring_result"]
        meta = s.get("metadata", {})
        candidate = os.path.basename(meta.get("resume_path", "candidate"))
        job_path = os.path.basename(meta.get("job_description_path", "job.pdf"))
        total_score = s.get("total_score", 0)
        criteria_requirements = meta.get("criteria_requirements", {})
        normalized_weights = meta.get("normalized_weights", {})
        gap = s.get("gap_analysis", {})
        missing_skills = gap.get("missing_skills", [])

        for criterion, data in s.items():
            if criterion in ("total_score", "metadata", "gap_analysis"):
                continue
            rec = {
                "candidate": candidate,
                "job": job_path,
                "criterion": criterion,
                "raw_score": data.get("raw_score", 0),
                "weight_given": data.get("weight_given", 0),
                "normalized_percentage": data.get("normalized_percentage", 0.0),
                "weighted_contribution": data.get("weighted_contribution", 0.0),
                "total_score": total_score,
                "missing_skills": ", ".join(missing_skills),
                "job_required_weight": criteria_requirements.get(criterion, 0),
                "job_normalized_weight_pct": normalized_weights.get(criterion, 0.0),
            }
            records.append(rec)
    return pd.DataFrame.from_records(records)


def render_bar_chart(df: pd.DataFrame, compare_mode: str = "group"):
    """Bar chart: per-criterion scores with flexible comparison modes.

    compare_mode: "group" (candidates as colors), "facet" (small multiples), "stack" (stacked).
    """
    if compare_mode == "facet":
        fig = px.bar(
            df,
            x="criterion",
            y="raw_score",
            color="criterion",
            facet_col="candidate",
            facet_col_wrap=3,
            hover_data=["normalized_percentage", "weighted_contribution", "missing_skills"],
            labels={"raw_score": "Score (0-100)", "criterion": "Criterion"},
            title="Per-criterion Scores by Candidate (Small Multiples)",
            template="plotly_white",
        )
    else:
        barmode = "group" if compare_mode == "group" else "stack"
        fig = px.bar(
            df,
            x="criterion",
            y="raw_score",
            color="candidate",
            barmode=barmode,
            hover_data=["normalized_percentage", "weighted_contribution", "missing_skills"],
            labels={"raw_score": "Score (0-100)", "criterion": "Criterion"},
            title="Per-criterion Scores (Click a bar for details)",
            template="plotly_white",
        )

    fig.update_layout(legend_title_text="Candidate", margin=dict(t=60, r=20, l=20, b=20))
    fig.update_xaxes(tickangle=0)
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{compare_mode}")
    return fig


def render_pie_chart(df: pd.DataFrame):
    # Pie: distribution of total score across factors for a single candidate
    fig = px.pie(
        df,
        names="criterion",
        values="weighted_contribution",
        title="Contribution to Total Score by Criterion",
        hover_data=["raw_score", "normalized_percentage"],
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, r=20, l=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="pie_chart")


def render_scatter(df: pd.DataFrame):
    # Scatter/line: candidate performance vs job benchmark (normalized weight %)
    fig = px.scatter(
        df,
        x="job_normalized_weight_pct",
        y="raw_score",
        color="criterion",
        hover_data=["candidate", "weighted_contribution"],
        labels={
            "job_normalized_weight_pct": "Job Weight % (Benchmark)",
            "raw_score": "Candidate Score (0-100)",
        },
        title="Candidate Performance vs Job Benchmark",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, r=20, l=20, b=20))
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True, key="scatter_chart")


def render_radar(df: pd.DataFrame):
    """Radar chart overlay: criteria as axes, candidates as traces."""
    criteria = sorted(df["criterion"].unique().tolist())
    fig = go.Figure()
    for cand, sub in df.groupby("candidate"):
        vals = [sub[sub["criterion"] == c]["raw_score"].mean() if not sub[sub["criterion"] == c].empty else 0 for c in criteria]
        fig.add_trace(
            go.Scatterpolar(r=vals, theta=criteria, fill="toself", name=cand, hovertemplate="%{theta}: %{r}<extra>" + cand + "</extra>")
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        template="plotly_white",
        title="Radar: Candidate Scores Across Criteria",
        margin=dict(t=60, r=20, l=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="radar_chart")


def render_heatmap(df: pd.DataFrame):
    """Heatmap: criteria x candidates matrix of scores for quick comparison."""
    mat = df.pivot_table(index="criterion", columns="candidate", values="raw_score", aggfunc="mean").fillna(0)
    fig = px.imshow(
        mat,
        color_continuous_scale="Blues",
        origin="lower",
        aspect="auto",
        labels=dict(color="Score"),
        title="Heatmap: Scores by Criterion and Candidate",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, r=20, l=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")


def render_line_chart(df: pd.DataFrame):
    """Line chart: per-candidate lines across criteria to show shape differences."""
    # Ensure a consistent criterion order on x-axis
    fig = px.line(
        df.sort_values(["candidate", "criterion"]),
        x="criterion",
        y="raw_score",
        color="candidate",
        markers=True,
        labels={"raw_score": "Score (0-100)", "criterion": "Criterion"},
        title="Line: Score Across Criteria",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=60, r=20, l=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="line_chart")


def render_parallel_coordinates(df: pd.DataFrame):
    """Parallel coordinates: criteria as dimensions, each candidate as a line."""
    # Pivot to wide: rows=candidate, cols=criterion
    wide = df.pivot_table(index="candidate", columns="criterion", values="raw_score", aggfunc="mean").fillna(0)
    if wide.shape[0] < 1 or wide.shape[1] < 2:
        st.info("Need at least 1 candidate and 2 criteria for parallel coordinates.")
        return
    dims = [
        dict(range=[0, 100], label=str(col), values=wide[col].values)
        for col in wide.columns
    ]
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=list(range(len(wide))), colorscale="Tealrose", showscale=False),
        dimensions=dims,
    ))
    fig.update_layout(title="Parallel Coordinates: Criteria Profiles", template="plotly_white", margin=dict(t=60, r=20, l=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="parallel_coords")


def render_box_violin(df: pd.DataFrame):
    """Box/Violin: distribution of scores per criterion (across selected candidates)."""
    col1, col2 = st.columns(2)
    with col1:
        fig_b = px.box(
            df,
            x="criterion",
            y="raw_score",
            points="all",
            color="criterion",
            title="Box: Score Distribution per Criterion",
            template="plotly_white",
        )
        st.plotly_chart(fig_b, use_container_width=True, key="boxplot")
    with col2:
        fig_v = px.violin(
            df,
            x="criterion",
            y="raw_score",
            box=True,
            points="all",
            color="criterion",
            title="Violin: Score Distribution per Criterion",
            template="plotly_white",
        )
        st.plotly_chart(fig_v, use_container_width=True, key="violin")


def render_sankey_dependency(df: pd.DataFrame):
    """Dependency graph via Sankey: Candidate -> Criterion with weighted contribution as flow."""
    # Build node list: candidates + criteria
    candidates = df["candidate"].unique().tolist()
    criteria = df["criterion"].unique().tolist()
    nodes = candidates + criteria
    node_index = {n: i for i, n in enumerate(nodes)}

    # Links: from candidate to criterion, value = weighted_contribution (fallback to raw_score)
    links = df.groupby(["candidate", "criterion"], observed=False).agg({
        "weighted_contribution": "sum",
        "raw_score": "mean",
    }).reset_index()
    values = links["weighted_contribution"].fillna(links["raw_score"]).clip(lower=0.0)
    source = links["candidate"].map(node_index)
    target = links["criterion"].map(node_index)

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="gray", width=0.5),
            label=nodes,
            color=["#1f77b4"] * len(candidates) + ["#ff7f0e"] * len(criteria),
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            hovertemplate="%{source.label} → %{target.label}<br>Value: %{value}<extra></extra>",
        ),
    )])
    fig.update_layout(title_text="Dependency Graph: Candidate ↔ Criterion Contributions", font_size=12, template="plotly_white", margin=dict(t=60, r=20, l=20, b=20))
    st.plotly_chart(fig, use_container_width=True, key="sankey")


def render_click_details(df: pd.DataFrame, click_data: Dict[str, Any]):
    if not click_data:
        st.info("Click a bar in the bar chart to see detailed comparison.")
        return
    pts = click_data.get("points", [])
    if not pts:
        st.info("Click a bar in the bar chart to see detailed comparison.")
        return
    point = pts[0]
    criterion_clicked = point.get("x")
    candidate_clicked = point.get("curveNumber")

    # Determine candidate name from trace index
    # Map trace index -> candidate in current figure
    # As fallback, filter by x and pick first row
    sub = df[df["criterion"] == criterion_clicked]
    if sub.empty:
        st.warning("No details found for selection.")
        return

    # If multiple candidates selected, show all rows for the criterion
    st.subheader(f"Details: {criterion_clicked}")
    show_cols = [
        "candidate",
        "criterion",
        "raw_score",
        "job_required_weight",
        "job_normalized_weight_pct",
        "normalized_percentage",
        "weighted_contribution",
        "missing_skills",
    ]
    st.dataframe(sub[show_cols].sort_values("candidate"), use_container_width=True)


def main():
    st.set_page_config(page_title="Resume Scoring Dashboard", layout="wide")
    st.title("Resume Scoring Dashboard")
    st.caption("Interactive view of candidate scores vs job criteria")

    with st.sidebar:
        st.header("Data")
        st.write("Pipeline output path:")
        st.code(OUTPUT_JSON, language="text")
        last_updated = get_file_last_updated(OUTPUT_JSON)
        st.caption(f"Last updated: {last_updated}")
        colr1, colr2 = st.columns(2)
        with colr1:
            refresh = st.button("Refresh data")
        with colr2:
            auto = st.toggle("Auto-refresh", value=False, help="Reload when file changes")

    # Detect file changes and auto-refresh
    current_sig = os.path.getmtime(OUTPUT_JSON) if os.path.exists(OUTPUT_JSON) else 0
    previous_sig = st.session_state.get("_file_sig", None)
    if auto and previous_sig is not None and current_sig != previous_sig:
        load_results.clear()
        st.session_state["_file_sig"] = current_sig
        st.experimental_rerun()
    st.session_state["_file_sig"] = current_sig

    results = load_results(OUTPUT_JSON)
    if refresh:
        load_results.clear()
        results = load_results(OUTPUT_JSON)

    if not results:
        st.warning("No results found. Run the pipeline to generate outputs/scoring_results.json")
        return

    long_df = results_to_long_df(results)
    if long_df.empty:
        st.warning("No successful scoring results to display.")
        return

    # Candidate filter(s)
    candidates = sorted(long_df["candidate"].unique().tolist())
    default_sel = candidates if len(candidates) <= 3 else candidates[:3]
    selected_candidates = st.multiselect(
        "Select candidate(s)", options=candidates, default=default_sel, help="Pick one or more candidates to compare"
    )
    filtered = long_df[long_df["candidate"].isin(selected_candidates)]

    # Criterion filter
    criteria = sorted(filtered["criterion"].unique().tolist())
    selected_criteria = st.multiselect(
        "Select criteria", options=criteria, default=criteria
    )
    filtered = filtered[filtered["criterion"].isin(selected_criteria)]

    # Sort criteria by job weight or variance across candidates
    sort_mode = st.radio(
        "Sort criteria by",
        options=["job weight", "variance"],
        horizontal=True,
        help="Reorder criteria to emphasize the most important or most different",
    )
    if sort_mode == "job weight":
        order = (
            filtered.groupby("criterion", as_index=False, observed=False)["job_required_weight"].mean().sort_values("job_required_weight", ascending=False)[
                "criterion"
            ].tolist()
        )
    else:
        order = (
            filtered.groupby("criterion", observed=False)["raw_score"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False).index.tolist()
        )
    filtered["criterion"] = pd.Categorical(filtered["criterion"], categories=order, ordered=True)

    # Tabs for views
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Comparison Bars",
        "Radar",
        "Heatmap",
        "Benchmark Scatter",
        "Line",
        "Distributions",
        "Dependency Graph",
    ])

    with t1:
        compare_mode = st.segmented_control("Bar mode", options=["group", "facet", "stack"], default="group")
        col1, col2 = st.columns([2, 1])
        with col1:
            bar_fig = render_bar_chart(filtered, compare_mode=compare_mode)
        with col2:
            if len(selected_candidates) == 1:
                render_pie_chart(filtered[filtered["candidate"] == selected_candidates[0]])
            else:
                agg = filtered.groupby("criterion", as_index=False, observed=False)["weighted_contribution"].sum()
                st.subheader("Aggregate Contribution (Selected Candidates)")
                fig = px.pie(
                    agg,
                    names="criterion",
                    values="weighted_contribution",
                    title="Contribution to Total Score (Aggregated)",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True, key="pie_chart_agg")

    with t2:
        render_radar(filtered)

    with t3:
        render_heatmap(filtered)

    with t4:
        render_scatter(filtered)

    with t5:
        render_line_chart(filtered)

    with t6:
        render_box_violin(filtered)

    with t7:
        render_sankey_dependency(filtered)

    with t5:
        st.subheader("Detailed Comparison")
        # Workaround: Streamlit doesn't natively capture plotly click; use events via
        try:
            from streamlit_plotly_events import plotly_events
            st.caption("Click a bar in the Bars tab; or select below if unavailable.")
            # Re-render a lightweight bar for click capture to avoid cross-tab issues
            mini = px.bar(
                filtered,
                x="criterion",
                y="raw_score",
                color="candidate",
                barmode="group",
                template="plotly_white",
            )
            events = plotly_events(mini, click_event=True, hover_event=False, select_event=False, override_height=0, override_width=0)
            click_data = {"points": events} if events else None
            render_click_details(filtered, click_data)
        except Exception:
            # Fallback simple selector
            st.caption("Extension 'streamlit-plotly-events' not installed. Using fallback selector.")
            sel_crit = st.selectbox("Select a criterion for details", options=criteria)
            render_click_details(filtered, {"points": [{"x": sel_crit}]})

    st.markdown("---")
    st.caption(
        "Tips: Use tabs to switch views. Sort criteria to highlight differences. Refresh after re-running pipeline."
    )


if __name__ == "__main__":
    main()


