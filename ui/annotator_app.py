"""Annotator UI — review queue sorted by uncertainty, highest first.

Run: streamlit run ui/annotator_app.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDIDATES_PATH = DATA_DIR / "labeled_candidates.csv"
FINAL_PATH = DATA_DIR / "final_labels.csv"

st.set_page_config(page_title="Review Queue : Active learning loop", layout="centered")


@st.cache_data
def load_candidates():
    df = pd.read_csv(CANDIDATES_PATH)
    return df.sort_values("uncertainty", ascending=False).reset_index(drop=True)


def load_final():
    if FINAL_PATH.exists():
        return pd.read_csv(FINAL_PATH)
    return pd.DataFrame(columns=["id", "text", "predicted_label", "final_label", "was_edited", "review_time_seconds"])


def save_decision(row, final_label, was_edited, review_time):
    final_df = load_final()
    final_df = final_df[final_df["id"] != row["id"]]  # replace if re-reviewed
    new_row = pd.DataFrame([{
        "id": row["id"], "text": row["text"], "predicted_label": row["predicted_label"],
        "final_label": final_label, "was_edited": was_edited, "review_time_seconds": round(review_time, 1),
    }])
    final_df = pd.concat([final_df, new_row], ignore_index=True)
    final_df.to_csv(FINAL_PATH, index=False)


def main():
    if not CANDIDATES_PATH.exists():
        st.error(f"No candidates found at {CANDIDATES_PATH}. Run scripts/label_batch.py first.")
        return

    candidates = load_candidates()
    final_df = load_final()
    reviewed_ids = set(final_df["id"]) if not final_df.empty else set()
    pending = candidates[~candidates["id"].isin(reviewed_ids)].reset_index(drop=True)

    st.title("Review Queue")
    st.caption("Sorted by uncertainty, highest first. Auto-confident rows go straight to final output.")

    total = len(candidates)
    done = len(reviewed_ids)
    st.progress(done / total if total else 0, text=f"{done}/{total} reviewed")

    if pending.empty:
        st.success("Queue is empty. Every row has been reviewed.")
        if st.button("Export auto-confident rows (uncertainty = 0) into final labels"):
            auto = candidates[candidates["uncertainty"] == 0]
            for _, row in auto.iterrows():
                if row["id"] not in reviewed_ids:
                    save_decision(row, row["predicted_label"], was_edited=False, review_time=0)
            st.rerun()
        return

    if "review_start" not in st.session_state:
        st.session_state.review_start = time.time()

    row = pending.iloc[0]

    st.markdown(f"**Text:** {row['text']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted", row["predicted_label"])
    col2.metric("Confidence", f"{row['confidence']:.0%}")
    col3.metric("Uncertainty", f"{row['uncertainty']:.2f}")

    if "cross_model_disagree" in row.index and row["cross_model_disagree"]:
        st.warning(
            f"Cross-model disagreement — **{row.get('model_a', 'Model A')}**: "
            f"{row['model_a_label']} ({row['model_a_votes']})  vs.  "
            f"**{row.get('model_b', 'Model B')}**: {row['model_b_label']} ({row['model_b_votes']})"
        )
    elif "model_a_votes" in row.index:
        st.caption(f"Model A votes: {row['model_a_votes']}  |  Model B votes: {row['model_b_votes']}")

    labels = sorted(candidates["predicted_label"].unique().tolist())
    choice = st.radio("Final label:", labels, index=labels.index(row["predicted_label"]))

    if st.button("Accept & Next", type="primary"):
        elapsed = time.time() - st.session_state.review_start
        was_edited = choice != row["predicted_label"]
        save_decision(row, choice, was_edited, elapsed)
        st.session_state.review_start = time.time()
        st.rerun()

    with st.expander("Run stats"):
        avg_time = final_df["review_time_seconds"].mean() if not final_df.empty else 0
        edit_rate = final_df["was_edited"].mean() if not final_df.empty else 0
        st.write(f"Avg review time: {avg_time:.1f}s | Edit rate: {edit_rate*100:.1f}%")


if __name__ == "__main__":
    main()