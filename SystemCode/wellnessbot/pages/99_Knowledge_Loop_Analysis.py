from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is on sys.path for Streamlit Cloud
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from SystemCode.wellnessbot.offline.mine_logs import run_mining

# --------------------------------------------
# PAGE TRACKING FOR CHAT RESET
# --------------------------------------------
CURRENT_PAGE = "mining"

if "active_page" not in st.session_state:
    st.session_state.active_page = None

if "force_chat_reset" not in st.session_state:
    st.session_state.force_chat_reset = False

previous_page = st.session_state.active_page
if previous_page is not None and previous_page != CURRENT_PAGE:
    st.session_state.force_chat_reset = True

st.session_state.active_page = CURRENT_PAGE

st.set_page_config(page_title="Knowledge Loop Analysis", layout="wide")
st.title("Knowledge Loop Analysis")
st.caption("Offline mining for interaction logs + feedback logs")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERACTIONS = PROJECT_ROOT / "logs"
DEFAULT_FEEDBACK = PROJECT_ROOT / "logs"

st.subheader("Input files")

col1, col2 = st.columns(2)

with col1:
    interactions_path_str = st.text_input(
        "Interactions JSONL path",
        value=str(DEFAULT_INTERACTIONS),
    )

with col2:
    feedback_path_str = st.text_input(
        "Feedback JSONL path",
        value=str(DEFAULT_FEEDBACK),
    )

run_clicked = st.button("Run mining", type="primary")

if run_clicked:
    try:
        interactions_path = Path(interactions_path_str)
        feedback_path = Path(feedback_path_str)

        results = run_mining(interactions_path, feedback_path)

        st.success("Mining complete.")

        # ----------------------------
        # Basic summary
        # ----------------------------
        st.subheader("Basic summary")
        stats = results["basic_stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Interactions", stats["interaction_count"])
        c2.metric("Matched feedback", stats["matched_feedback_count"])
        c3.metric("Unmatched feedback", stats["unmatched_feedback_count"])
        c4.metric("Candidate cases", len(results["candidate_cases"]))

        # ----------------------------
        # 1. Candidate cases
        # ----------------------------
        st.subheader("Candidate cases")
        cases_df = pd.DataFrame(results["candidate_cases"])
        if not cases_df.empty:
            st.dataframe(cases_df, use_container_width=True, hide_index=True)
        else:
            st.info("No candidate cases generated.")

        # Clear separation: candidate cases vs data distribution
        st.divider()

        # ----------------------------
        # 2. KG rule mining (rule combination)
        # ----------------------------
        st.subheader("KG rule mining")
        combo_df = pd.DataFrame(results["rule_combination_summary"])
        if not combo_df.empty:
            st.dataframe(combo_df, use_container_width=True, hide_index=True)
        else:
            st.info("No combined rule patterns found.")

        # ----------------------------
        # 3. Safety consistency (red flag)
        # ----------------------------
        st.subheader("Safety consistency")
        redflag_df = pd.DataFrame(results["red_flag_consistency"])
        if not redflag_df.empty:
            st.dataframe(redflag_df, use_container_width=True, hide_index=True)
        else:
            st.info("No red-flag cases found.")

        # ----------------------------
        # 4. Planner mining (selection + feedback)
        # ----------------------------
        st.subheader("Planner mining")
        planner_df = pd.DataFrame(results["planner_selection_summary"])
        if not planner_df.empty:
            st.dataframe(planner_df, use_container_width=True, hide_index=True)
        else:
            st.info("No planner selection data found.")

        # ----------------------------
        # Optional raw outputs
        # ----------------------------
        with st.expander("Show unmatched feedback JSON"):
            st.code(json.dumps(results["unmatched_feedback"], indent=2), language="json")

        with st.expander("Show raw mining JSON"):
            st.code(json.dumps(results, indent=2), language="json")

    except Exception as e:
        st.error(f"Mining failed: {e}")