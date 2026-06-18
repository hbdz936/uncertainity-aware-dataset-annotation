"""Batch-label a CSV using self-consistency sampling across two independent
models, producing per-row predicted label, confidence, uncertainty, and an
explicit cross-model disagreement flag for active-learning triage.

Why two models: self-consistency (resampling one model) only catches
stochastic uncertainty. A model with a confident-but-wrong prior gives the
same wrong answer every time, so single-model entropy stays at 0. Sampling
a second, independent model and pooling its votes surfaces that case too.

Usage:
    python scripts/label_batch.py --input data/input.csv --out data/labeled_candidates.csv \
        --labels positive,negative,neutral --k 3
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.llm_client import GroqLabelClient, DEFAULT_MODEL
from core.uncertainty import majority_vote, cross_model_disagreement

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "runs.db"


def log_run(run_id: str, model_a: str, model_b: str, n_rows: int, k: int,
            total_cost: float, elapsed_s: float, n_disagreements: int):
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT, model_a TEXT, model_b TEXT, n_rows INTEGER, k INTEGER,
            total_cost_usd REAL, elapsed_seconds REAL, n_disagreements INTEGER,
            timestamp TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (run_id, model_a, model_b, n_rows, k, total_cost, elapsed_s, n_disagreements),
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", required=True, help="comma-separated label set")
    parser.add_argument("--k", type=int, default=3, help="self-consistency samples PER model")
    parser.add_argument("--model-a", default=DEFAULT_MODEL, help="primary model")
    parser.add_argument("--model-b", default="llama-3.1-8b-instant",
                         help="second, independent model for cross-checking")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    labels = [l.strip() for l in args.labels.split(",")]
    df = pd.read_csv(args.input)
    client_a = GroqLabelClient(model=args.model_a)
    client_b = GroqLabelClient(model=args.model_b)

    results = []
    total_cost = 0.0
    n_disagreements = 0
    start = time.time()

    for i, row in df.iterrows():
        votes_a, votes_b = [], []
        row_cost = 0.0

        for _ in range(args.k):
            r = client_a.classify(row["text"], labels, temperature=args.temperature)
            votes_a.append(r.label)
            row_cost += r.cost_usd

        for _ in range(args.k):
            r = client_b.classify(row["text"], labels, temperature=args.temperature)
            votes_b.append(r.label)
            row_cost += r.cost_usd

        label_a, conf_a, entropy_a = majority_vote(votes_a)
        label_b, conf_b, entropy_b = majority_vote(votes_b)
        disagree = cross_model_disagreement(label_a, label_b)
        n_disagreements += disagree

        # Pool both models' votes into one distribution: catches stochastic
        # wavering AND systematic cross-model disagreement in one number.
        label, confidence, uncertainty = majority_vote(votes_a + votes_b)
        total_cost += row_cost

        results.append({
            "id": row["id"],
            "text": row["text"],
            "predicted_label": label,
            "confidence": round(confidence, 3),
            "uncertainty": round(uncertainty, 3),
            "model_a": args.model_a,
            "model_a_label": label_a,
            "model_a_votes": "|".join(votes_a),
            "model_b": args.model_b,
            "model_b_label": label_b,
            "model_b_votes": "|".join(votes_b),
            "cross_model_disagree": disagree,
            "cost_usd": round(row_cost, 6),
        })

        flag = "  <-- CROSS-MODEL DISAGREEMENT" if disagree else ""
        print(f"[{i+1}/{len(df)}] id={row['id']} -> {label} "
              f"(conf={confidence:.2f}, unc={uncertainty:.2f}){flag}")

    out_df = pd.DataFrame(results).sort_values("uncertainty", ascending=False)
    out_df.to_csv(args.out, index=False)

    elapsed = time.time() - start
    log_run(run_id=str(int(start)), model_a=args.model_a, model_b=args.model_b,
             n_rows=len(df), k=args.k, total_cost=total_cost, elapsed_s=elapsed,
             n_disagreements=n_disagreements)

    auto_labeled = (out_df["uncertainty"] == 0).sum()
    print(f"\nDone. {len(out_df)} rows -> {args.out}")
    print(f"Coverage (zero uncertainty across BOTH models, no review needed): "
          f"{auto_labeled}/{len(out_df)} ({100*auto_labeled/len(out_df):.1f}%)")
    print(f"Cross-model disagreements: {n_disagreements}/{len(out_df)} rows "
          f"where {args.model_a} and {args.model_b} picked different majority labels")
    print(f"Total cost: ${total_cost:.4f} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()