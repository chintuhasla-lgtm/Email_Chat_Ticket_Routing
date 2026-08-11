"""05_multilabel_departments.py
Create a synthetic multi-label department target, train a One-vs-Rest
LogisticRegression classifier on TF-IDF features, and save the model.

Steps:
- load `data/cleaned_tickets.csv`
- construct binary indicator matrix combining primary `Ticket_Type`
  with inferred secondary departments from keyword co-occurrence
- train OneVsRestClassifier(LogisticRegression) on TF-IDF features
- predict probabilities and assign departments with prob >= 0.3
- report count/percentage of multi-department test tickets
- save the fitted model to `multilabel_model.pkl`
"""
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import joblib


def infer_secondary_departments(text: str, departments_keywords: dict) -> set:
    """Return a set of department labels inferred from keyword overlap in text."""
    labels = set()
    if not isinstance(text, str) or not text:
        return labels
    t = text.lower()
    for dept, keywords in departments_keywords.items():
        for kw in keywords:
            if kw in t:
                labels.add(dept)
                break
    return labels


def build_indicator_matrix(df: pd.DataFrame, departments: list, keywords_map: dict) -> pd.DataFrame:
    """Construct a binary indicator DataFrame for given department list.

    Each row receives a 1 for the primary `Ticket_Type` and any inferred
    secondary departments from `keywords_map`.
    """
    indicators = []
    for _, row in df.iterrows():
        present = set()
        primary = row.get("Ticket_Type")
        if isinstance(primary, str) and primary:
            present.add(primary)
        inferred = infer_secondary_departments(row.get("clean_text", ""), keywords_map)
        present.update(inferred)
        indicators.append([1 if dept in present else 0 for dept in departments])
    return pd.DataFrame(indicators, columns=departments)


def main():
    # Load cleaned tickets
    try:
        df = pd.read_csv("data/cleaned_tickets.csv")
    except FileNotFoundError:
        print("Error: data/cleaned_tickets.csv not found. Run 02_text_cleaning.py first.")
        sys.exit(1)

    # Define departments (use unique primary labels plus common others)
    primary_depts = sorted(df["Ticket_Type"].dropna().unique().tolist())
    # Ensure common queues are present
    extra = [d for d in ["Billing", "Technical", "General"] if d not in primary_depts]
    departments = primary_depts + extra

    # Keyword map to infer secondary departments
    keywords_map = {
        "Billing": ["invoice", "billing", "refund", "charge", "payment"],
        "Technical": ["error", "bug", "api", "login", "crash", "database", "slow", "not working", "down"],
        "General": ["hours", "location", "upgrade", "question", "info", "support"],
    }

    # Build multi-label indicator matrix
    Y = build_indicator_matrix(df, departments, keywords_map)

    # Keep samples that have at least one positive label
    mask = Y.sum(axis=1) > 0
    X_all = df.loc[mask, "clean_text"].astype(str)
    Y_all = Y.loc[mask].reset_index(drop=True)

    if X_all.empty:
        print("No samples with any department labels found after synthetic labeling.")
        sys.exit(1)

    # Train/test split (no stratification because target is multi-label)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )

    # Build pipeline: TF-IDF -> OneVsRest(LogisticRegression)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    # Fit TF-IDF on training text and transform
    X_train_tfidf = tfidf.fit_transform(X_train)
    clf.fit(X_train_tfidf, Y_train.values)

    # Save a combined object (vectorizer + classifier) for convenience
    multilabel_pipeline = {"vectorizer": tfidf, "classifier": clf, "departments": departments}
    joblib.dump(multilabel_pipeline, "multilabel_model.pkl")
    print("Saved multilabel model to multilabel_model.pkl")

    # Predict probabilities on test set
    X_test_tfidf = tfidf.transform(X_test)
    try:
        probs = clf.predict_proba(X_test_tfidf)
    except AttributeError:
        # Some sklearn versions may not implement predict_proba for OneVsRestClassifier with this estimator
        # Fallback: use decision_function and map via logistic sigmoid
        dec = clf.decision_function(X_test_tfidf)
        probs = 1 / (1 + np.exp(-dec))

    # Assign departments with threshold 0.3
    threshold = 0.3
    assigned = (probs >= threshold).astype(int)

    # Count multi-department tickets (more than one assigned department)
    multi_counts = (assigned.sum(axis=1) > 1).sum()
    total = assigned.shape[0]
    pct = 100.0 * multi_counts / total if total > 0 else 0.0

    print(f"Test tickets: {total}")
    print(f"Multi-department flagged: {multi_counts} ({pct:.2f}%)")

    # Optionally show an example of assignments (first 5)
    for i in range(min(5, total)):
        probs_i = probs[i]
        assigned_depts = [departments[j] for j, v in enumerate(assigned[i]) if v]
        print(f"\nSample {i}: Assigned -> {assigned_depts}")
        print(f"Probs: {dict(zip(departments, [round(float(p),3) for p in probs_i]))}")


if __name__ == "__main__":
    main()
