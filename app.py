"""
Interactive Process Unlearning Dashboard
Choose a process to delete, run unlearning, and view metrics.
"""

import streamlit as st
from core import run_unlearning_pipeline

st.set_page_config(
    page_title="Process Unlearning",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Custom styles ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .metric-card h4 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #22d3ee;
    }
    
    .success-value { color: #34d399 !important; }
    .warning-value { color: #fbbf24 !important; }
    
    .stSelectbox > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    .run-btn {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22d3ee, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="header-title">Process Unlearning Dashboard</p>', unsafe_allow_html=True)
st.markdown("Choose a process to **delete/unlearn** from the model and view the metrics below.")
st.divider()

# ---- Sidebar: Process selection ----
with st.sidebar:
    st.markdown("### Settings")
    target_process = st.selectbox(
        "Process to unlearn",
        options=[f"P{i}" for i in range(10)],
        index=3,
        help="Select the process you want to remove from the model",
    )
    st.markdown("---")
    st.caption("Processes P0–P9 are synthetic memory traces. Unlearning degrades the model's ability to predict the selected process.")

# ---- Run button ----
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    run_clicked = st.button("Unlearn process", type="primary", use_container_width=True)

if run_clicked:
    progress_placeholder = st.empty()
    with st.status(f"Unlearning **{target_process}** — Training & erasing...", expanded=True) as status:
        def log(msg):
            st.write(msg)

        result = run_unlearning_pipeline(target_process, log_fn=log)
        status.update(label="Done", state="complete")

    # ---- Certified Forgetting Score (hero metric) ----
    st.markdown("## Certified Forgetting Score")
    cfs = result["certified_forgetting_score"]
    st.markdown(f"""
    <div class="metric-card">
        <h4>RMSE increase on target process ({target_process})</h4>
        <span class="value {'success-value' if cfs > 0 else 'warning-value'}">{cfs:.4f}</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Higher = more effective forgetting. Target process predictions degrade while others stay stable.")

    # ---- Metrics tables ----
    st.markdown("## Metrics")

    tab1, tab2, tab3 = st.tabs(["Baseline", "After unlearning", "Comparison"])

    with tab1:
        st.dataframe(result["baseline_metrics"], use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(result["after_metrics"], use_container_width=True, hide_index=True)

    with tab3:
        st.dataframe(result["comparison_df"], use_container_width=True, hide_index=True)

    # Target process highlight
    st.markdown("---")
    target_row = result["comparison_df"][result["comparison_df"]["Process"] == target_process].iloc[0]
    st.markdown(f"""
    **Target process {target_process}:** Before RMSE = `{target_row['Before RMSE']}` → After RMSE = `{target_row['After RMSE']}`  
    (R² before: `{target_row['Before R2']}` → after: `{target_row['After R2']}`)
    """)
else:
    st.info("Select a process and click **Unlearn process** to run the pipeline.")
