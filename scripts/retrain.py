"""Train a small downstream classifier (TF-IDF + Logistic Regression) on the
human-finalized labels, to demonstrate the labeled dataset is actually useful
for retraining — not just an annotation exercise.

Usage:
    python scripts/retrain.py --data data/final_labels.csv
"""
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label-col", default="final_label")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.label_col not in df.columns:
        args.label_col = "predicted_label"

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[args.label_col], test_size=0.25, random_state=42, stratify=df[args.label_col]
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"Downstream classifier (TF-IDF + LogReg) on {len(df)} labeled rows:")
    print(f"  Accuracy: {acc:.3f} | Macro F1: {f1:.3f}")
    print("\nRun again after each review batch to show accuracy improving with more reviewed data.")


if __name__ == "__main__":
    main()