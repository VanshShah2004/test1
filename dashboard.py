#!/usr/bin/env python3
"""
Professional Resume Scoring Dashboard (Streamlit)
Light Mode | Interactive | Easy to Use

Run: streamlit run dashboard.py
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Custom CSS for professional light mode design
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f2937;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 600;
        font-size: 1.25rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Selectbox and multiselect styling */
    .stSelectbox label, .stMultiSelect label {
        font-weight: 600;
        color: #374151;
        font-size: 0.95rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #6b7280;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6;
        background-color: #eff6ff;
    }
</style>
""", unsafe_allow_html=True)

OUTPUT_JSON = os.path.join("outputs", "scoring_results.json")

# Light mode color palette
COLORS = {
    'primary': '#3b82f6',
    'secondary': '#6366f1',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'background': '#ffffff',
    'surface': '#f9fafb',
    'text_primary': '#1f2937',
    'text_secondary': '#6b7280',
    'border': '#e5e7eb',
}

@st.cache_data(show_spinner=False)
def load_results(path: str) -> List[Dict[str, Any]]:
    """Load scoring results from JSON file."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_file_last_updated(path: str) -> str:
    """Get last modified timestamp of file."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown"


def results_to_long_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert nested results to long format DataFrame."""
    records: List[Dict[str, Any]] = []
    
    if not isinstance(results, list):
        return pd.DataFrame()
    
    for r in results:
        try:
            if not isinstance(r, dict) or not r.get("success"):
                continue
            
            # Safely get scoring_result
            s = r.get("scoring_result", {})
            if not isinstance(s, dict):
                continue
            
            meta = s.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            
            candidate = os.path.basename(meta.get("resume_path", "candidate")).replace(".pdf", "")
            job_path = os.path.basename(meta.get("job_description_path", "job.pdf")).replace(".pdf", "")
            total_score = s.get("total_score", 0)
            criteria_requirements = meta.get("criteria_requirements", {})
            if not isinstance(criteria_requirements, dict):
                criteria_requirements = {}
            normalized_weights = meta.get("normalized_weights", {})
            if not isinstance(normalized_weights, dict):
                normalized_weights = {}
            gap = s.get("gap_analysis", {})
            if not isinstance(gap, dict):
                gap = {}
            missing_skills = gap.get("missing_skills", [])
            if not isinstance(missing_skills, list):
                missing_skills = []
            
            # Get hybrid scoring metadata
            llm_score = meta.get("llm_total_score")
            slm_score = meta.get("slm_total_score")
            scoring_method = meta.get("scoring_method", "unknown")

            # Iterate through scoring result items
            for criterion, data in s.items():
                # Skip non-criteria keys
                if criterion in ("total_score", "metadata", "gap_analysis"):
                    continue
                
                # Ensure data is a dictionary
                if not isinstance(data, dict):
                    continue
                
                # Safely extract values with proper type conversion and defaults
                try:
                    raw_score_val = data.get("raw_score")
                    raw_score = float(raw_score_val) if raw_score_val is not None else 0.0
                except (ValueError, TypeError):
                    raw_score = 0.0
                
                try:
                    weight_given_val = data.get("weight_given")
                    weight_given = int(weight_given_val) if weight_given_val is not None else 0
                except (ValueError, TypeError):
                    weight_given = 0
                
                try:
                    norm_pct_val = data.get("normalized_percentage")
                    normalized_percentage = float(norm_pct_val) if norm_pct_val is not None else 0.0
                except (ValueError, TypeError):
                    normalized_percentage = 0.0
                
                try:
                    weighted_contrib_val = data.get("weighted_contribution")
                    weighted_contribution = float(weighted_contrib_val) if weighted_contrib_val is not None else 0.0
                except (ValueError, TypeError):
                    weighted_contribution = 0.0
                
                try:
                    total_score_val = float(total_score) if total_score is not None else 0.0
                except (ValueError, TypeError):
                    total_score_val = 0.0
                
                try:
                    job_weight = int(criteria_requirements.get(criterion, 0)) if criteria_requirements.get(criterion) is not None else 0
                except (ValueError, TypeError):
                    job_weight = 0
                
                try:
                    job_norm_weight = float(normalized_weights.get(criterion, 0.0)) if normalized_weights.get(criterion) is not None else 0.0
                except (ValueError, TypeError):
                    job_norm_weight = 0.0
                
                try:
                    llm_score_float = float(llm_score) if llm_score is not None else None
                except (ValueError, TypeError):
                    llm_score_float = None
                
                try:
                    slm_score_float = float(slm_score) if slm_score is not None else None
                except (ValueError, TypeError):
                    slm_score_float = None
                
                rec = {
                    "candidate": str(candidate),
                    "job": str(job_path),
                    "criterion": str(criterion).replace("_", " ").title(),
                    "raw_score": raw_score,
                    "weight_given": weight_given,
                    "normalized_percentage": normalized_percentage,
                    "weighted_contribution": weighted_contribution,
                    "total_score": total_score_val,
                    "missing_skills": ", ".join([str(s) for s in missing_skills]) if missing_skills else "None",
                    "job_required_weight": job_weight,
                    "job_normalized_weight_pct": job_norm_weight,
                    "llm_score": llm_score_float,
                    "slm_score": slm_score_float,
                    "scoring_method": str(scoring_method) if scoring_method else "unknown",
                }
                records.append(rec)
        
        except Exception as e:
            # Skip malformed records and continue
            continue
    
    if not records:
        return pd.DataFrame()
    
    try:
        return pd.DataFrame.from_records(records)
    except Exception:
        return pd.DataFrame()


def render_summary_metrics(df: pd.DataFrame):
    """Render summary metrics at the top."""
    if df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df.groupby("candidate")["total_score"].mean().mean()
        st.metric(
            label="Average Score",
            value=f"{avg_score:.1f}",
            delta=f"{avg_score - 70:.1f} vs 70 baseline" if avg_score >= 70 else None,
        )
    
    with col2:
        total_candidates = df["candidate"].nunique()
        st.metric(
            label="Candidates",
            value=total_candidates,
        )
    
    with col3:
        total_criteria = df["criterion"].nunique()
        st.metric(
            label="Criteria",
            value=total_criteria,
        )
    
    with col4:
        top_score = df["total_score"].max()
        st.metric(
            label="Highest Score",
            value=f"{top_score:.1f}",
        )


def render_bar_chart(df: pd.DataFrame, compare_mode: str = "group"):
    """Bar chart with professional styling."""
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
            title="Per-criterion Scores by Candidate",
            color_discrete_sequence=px.colors.qualitative.Set3,
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
            title="Per-criterion Scores Comparison",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )

    fig.update_layout(
        legend_title_text="Candidate",
        margin=dict(t=40, r=20, l=20, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937', size=12),
        title_font=dict(size=18, color='#1f2937'),
    )
    fig.update_xaxes(
        tickangle=-45,
        showgrid=True,
        gridcolor='#e5e7eb',
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#e5e7eb',
        range=[0, 110],
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#10b981", annotation_text="Good (80+)")
    fig.add_hline(y=60, line_dash="dash", line_color="#f59e0b", annotation_text="Average (60+)")
    st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{compare_mode}")


def render_pie_chart(df: pd.DataFrame):
    """Pie chart with professional styling."""
    fig = px.pie(
        df,
        names="criterion",
        values="weighted_contribution",
        title="Contribution to Total Score by Criterion",
        hover_data=["raw_score", "normalized_percentage"],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        title_font=dict(size=16, color='#1f2937'),
        margin=dict(t=40, r=20, l=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="pie_chart")


def render_scatter(df: pd.DataFrame):
    """Scatter plot with professional styling."""
    fig = px.scatter(
        df,
        x="job_normalized_weight_pct",
        y="raw_score",
        color="candidate",
        size="weighted_contribution",
        hover_data=["criterion", "weighted_contribution"],
        labels={
            "job_normalized_weight_pct": "Job Weight % (Importance)",
            "raw_score": "Candidate Score (0-100)",
        },
        title="Performance vs Job Requirements",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        title_font=dict(size=16, color='#1f2937'),
        margin=dict(t=40, r=20, l=20, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', range=[0, 110])
    fig.add_hline(y=80, line_dash="dash", line_color="#10b981")
    st.plotly_chart(fig, use_container_width=True, key="scatter_chart")


def render_radar(df: pd.DataFrame):
    """Radar chart with professional styling."""
    criteria = sorted(df["criterion"].unique().tolist())
    fig = go.Figure()
    
    colors = px.colors.qualitative.Pastel
    for idx, (cand, sub) in enumerate(df.groupby("candidate")):
        vals = [
            sub[sub["criterion"] == c]["raw_score"].mean() 
            if not sub[sub["criterion"] == c].empty else 0 
            for c in criteria
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=criteria,
                fill="toself",
                name=cand,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra>%{fullData.name}</extra>"
            )
        )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showgrid=True,
                gridcolor='#e5e7eb',
            )
        ),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        title_font=dict(size=16, color='#1f2937'),
        title="Multi-Criteria Performance Radar",
        margin=dict(t=40, r=20, l=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="radar_chart")


def render_heatmap(df: pd.DataFrame):
    """Heatmap with professional styling."""
    mat = df.pivot_table(
        index="criterion",
        columns="candidate",
        values="raw_score",
        aggfunc="mean"
    ).fillna(0)
    
    fig = px.imshow(
        mat,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="Score", x="Candidate", y="Criterion"),
        title="Score Heatmap: Candidates × Criteria",
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        title_font=dict(size=16, color='#1f2937'),
        margin=dict(t=40, r=20, l=20, b=60),
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")


def render_line_chart(df: pd.DataFrame):
    """Line chart with professional styling."""
    fig = px.line(
        df.sort_values(["candidate", "criterion"]),
        x="criterion",
        y="raw_score",
        color="candidate",
        markers=True,
        line_shape="spline",
        labels={"raw_score": "Score (0-100)", "criterion": "Criterion"},
        title="Score Trends Across Criteria",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f2937'),
        title_font=dict(size=16, color='#1f2937'),
        margin=dict(t=40, r=20, l=20, b=60),
    )
    fig.update_xaxes(tickangle=-45, showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', range=[0, 110])
    st.plotly_chart(fig, use_container_width=True, key="line_chart")


def render_comparison_table(df: pd.DataFrame):
    """Render detailed comparison table."""
    summary_df = df.groupby(["candidate", "criterion"]).agg({
        "raw_score": "mean",
        "weighted_contribution": "sum",
        "job_normalized_weight_pct": "first",
    }).reset_index()
    
    pivot_score = summary_df.pivot(
        index="criterion",
        columns="candidate",
        values="raw_score"
    ).round(1)
    
    st.dataframe(
        pivot_score,
        use_container_width=True,
        height=400,
    )


def render_hybrid_scoring_info(df: pd.DataFrame):
    """Display hybrid scoring information if available."""
    if "scoring_method" not in df.columns:
        return
    
    hybrid_rows = df[df["scoring_method"].str.contains("hybrid", case=False, na=False)]
    if hybrid_rows.empty:
        return
    
    st.subheader("📊 Scoring Methodology")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method = hybrid_rows["scoring_method"].iloc[0] if not hybrid_rows.empty else "Standard"
        st.info(f"**Method:** {method.replace('_', ' ').title()}")
    
    with col2:
        llm_avg = hybrid_rows["llm_score"].mean() if "llm_score" in hybrid_rows.columns else None
        if llm_avg and not pd.isna(llm_avg):
            st.metric("LLM Avg Score", f"{llm_avg:.1f}")
    
    with col3:
        slm_avg = hybrid_rows["slm_score"].mean() if "slm_score" in hybrid_rows.columns else None
        if slm_avg and not pd.isna(slm_avg):
            st.metric("SLM Avg Score", f"{slm_avg:.1f}")


def main():
    # Page config
    st.set_page_config(
        page_title="Resume Scoring Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Header
    st.title("📊 Resume Scoring Dashboard")
    st.markdown(
        """
        <p style='font-size: 1.1rem; color: #6b7280; margin-top: -1rem;'>
        Interactive analysis of candidate performance against job requirements
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("📁 Data Source")
        st.code(OUTPUT_JSON, language="text")
        last_updated = get_file_last_updated(OUTPUT_JSON)
        st.caption(f"📅 Last updated: {last_updated}")
        
        col1, col2 = st.columns(2)
        with col1:
            refresh = st.button("🔄 Refresh", use_container_width=True)
        with col2:
            auto = st.toggle("Auto-refresh", value=False, help="Reload when file changes")
        
        st.divider()
        
        st.subheader("ℹ️ About")
        st.markdown("""
        This dashboard provides interactive visualization 
        of resume scoring results from the hybrid LLM+SLM 
        scoring system.
        
        **Features:**
        - Multiple visualization types
        - Interactive filtering
        - Detailed metrics
        - Comparison tools
        """)
    
    # Auto-refresh logic
    current_sig = os.path.getmtime(OUTPUT_JSON) if os.path.exists(OUTPUT_JSON) else 0
    previous_sig = st.session_state.get("_file_sig", None)
    if auto and previous_sig is not None and current_sig != previous_sig:
        load_results.clear()
        st.session_state["_file_sig"] = current_sig
        st.rerun()
    st.session_state["_file_sig"] = current_sig
    
    # Load data
    try:
        results = load_results(OUTPUT_JSON)
        if refresh:
            load_results.clear()
            results = load_results(OUTPUT_JSON)
    except Exception as e:
        st.error(f"❌ Error loading results file: {e}")
        st.info(f"Please ensure the file exists at: `{OUTPUT_JSON}`")
        return
    
    # Check if data exists
    if not results:
        st.warning(
            "⚠️ No results found. Please run the pipeline to generate scoring results.\n\n"
            f"Expected file: `{OUTPUT_JSON}`"
        )
        return
    
    # Convert to DataFrame
    try:
        long_df = results_to_long_df(results)
    except Exception as e:
        st.error(f"❌ Error processing results data: {e}")
        st.info("The data format may be unexpected. Please check the scoring results file structure.")
        if st.checkbox("Show error details"):
            st.exception(e)
        return
    
    if long_df.empty:
        st.warning("⚠️ No successful scoring results to display.")
        return
    
    # Summary metrics
    st.markdown("---")
    render_summary_metrics(long_df)
    st.markdown("---")
    
    # Filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        candidates = sorted(long_df["candidate"].unique().tolist())
        default_sel = candidates if len(candidates) <= 3 else candidates[:3]
        selected_candidates = st.multiselect(
            "👥 Select Candidates",
            options=candidates,
            default=default_sel,
            help="Choose one or more candidates to compare",
        )
    
    with col2:
        criteria = sorted(long_df["criterion"].unique().tolist())
        selected_criteria = st.multiselect(
            "📋 Select Criteria",
            options=criteria,
            default=criteria[:min(5, len(criteria))],
            help="Filter by scoring criteria",
        )
    
    # Filter data
    filtered = long_df[
        (long_df["candidate"].isin(selected_candidates)) &
        (long_df["criterion"].isin(selected_criteria))
    ]
    
    if filtered.empty:
        st.info("ℹ️ No data matches the selected filters. Please adjust your selection.")
        return
    
    # Sort criteria
    sort_mode = st.radio(
        "🔀 Sort Criteria By",
        options=["Job Weight", "Variance", "Alphabetical"],
        horizontal=True,
        help="Reorder criteria for better analysis",
    )
    
    if sort_mode == "Job Weight":
        order = (
            filtered.groupby("criterion", as_index=False)["job_required_weight"]
            .mean()
            .sort_values("job_required_weight", ascending=False)["criterion"]
            .tolist()
        )
    elif sort_mode == "Variance":
        order = (
            filtered.groupby("criterion")["raw_score"]
            .agg(lambda s: s.max() - s.min())
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = sorted(filtered["criterion"].unique().tolist())
    
    filtered["criterion"] = pd.Categorical(filtered["criterion"], categories=order, ordered=True)
    
    # Hybrid scoring info
    render_hybrid_scoring_info(filtered)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Comparison",
        "🎯 Radar Analysis",
        "🔥 Heatmap",
        "📉 Trends",
        "📋 Detailed Table",
    ])
    
    with tab1:
        st.subheader("Score Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            compare_mode = st.radio(
                "Chart Style",
                options=["Group", "Stack", "Facet"],
                horizontal=True,
                key="overview_mode",
            )
            render_bar_chart(filtered, compare_mode=compare_mode.lower())
        
        with col2:
            if len(selected_candidates) == 1:
                render_pie_chart(filtered[filtered["candidate"] == selected_candidates[0]])
            else:
                st.info("💡 Select a single candidate to view contribution breakdown")
    
    with tab2:
        st.subheader("Performance Comparison")
        render_scatter(filtered)
        st.markdown("---")
        render_comparison_table(filtered)
    
    with tab3:
        st.subheader("Multi-Criteria Radar Analysis")
        render_radar(filtered)
    
    with tab4:
        st.subheader("Score Heatmap")
        render_heatmap(filtered)
    
    with tab5:
        st.subheader("Score Trends Across Criteria")
        render_line_chart(filtered)
    
    with tab6:
        st.subheader("Detailed Comparison Table")
        render_comparison_table(filtered)
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="scoring_results.csv",
            mime="text/csv",
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6b7280; padding: 1rem;'>
            <p>💡 <strong>Tips:</strong> Use filters to focus on specific candidates or criteria. 
            Click on charts for detailed information. Refresh data after re-running the pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
