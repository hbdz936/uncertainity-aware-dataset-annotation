"""Compare final (human-reviewed) labels against a gold subset.

Usage:
    python scripts/evaluate.py --gold data/gold.csv --pred data/final_labels.csv
"""
import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    merged = gold.merge(pred, on="id", suffixes=("_gold", "_pred"))
    if merged.empty:
        print("No overlapping ids between gold and predictions — check 'id' columns.")
        return

    y_true = merged["gold_label"]
    pred_col = "final_label" if "final_label" in merged.columns else "predicted_label"
    y_pred = merged[pred_col]

    print(f"Evaluated on {len(merged)} gold examples\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"Macro Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    if "was_edited" in merged.columns:
        edit_rate = merged["was_edited"].mean()
        print(f"\nHuman edit rate: {edit_rate*100:.1f}% "
              f"(i.e. {100-edit_rate*100:.1f}% of LLM labels accepted as-is)")


if __name__ == "__main__":
    main()