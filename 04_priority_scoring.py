"""04_priority_scoring.py
Score ticket priority using rule-based urgency signals combined with
a trained classifier fallback. Saves the trained priority model.

Produces a `predicted_priority` column and prints accuracy and a
confusion matrix comparing predictions to ground-truth `Ticket_Priority`.
"""
import re
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def rule_based_priority(clean_text: str) -> str:
    """Return a priority label from simple urgency keywords or None.

    Maps strong urgency words to 'High'/'Critical' heuristically.
    """
    if not isinstance(clean_text, str) or not clean_text.strip():
        return None

    txt = clean_text.lower()

    # Critical signals
    if "critical" in txt:
        return "Critical"

    # Urgent/high-priority keywords
    high_keywords = [
        "urgent",
        "immediately",
        "asap",
        "not working",
        "not working",
        "down",
        "immediately",
        "urgent",
    ]
    for kw in high_keywords:
        if kw in txt:
            return "High"

    # Angry / cancellation signals -> treat as High
    if any(w in txt for w in ["angry", "cancel", "sue", "complaint"]):
        return "High"

    # No rule-based signal found
    return None


def score_priority(clean_text: str, clf_pipeline: Pipeline) -> str:
    """Combine rule-based signal with classifier fallback to predict priority.

    If the rule-based method returns a label, use it; otherwise use the
    trained classifier pipeline to predict a priority label.
    """
    rule = rule_based_priority(clean_text)
    if rule is not None:
        return rule
    # fallback to classifier prediction
    return clf_pipeline.predict([clean_text])[0]


def main():
    # Load cleaned tickets
    try:
        df = pd.read_csv("data/cleaned_tickets.csv")
    except FileNotFoundError:
        print("Error: data/cleaned_tickets.csv not found. Run 02_text_cleaning.py first.")
        sys.exit(1)

    # Use only rows with non-empty clean_text
    df["clean_text"] = df["clean_text"].fillna("")

    # Prepare training data for priority classifier (drop rows without label)
    labeled = df.dropna(subset=["Ticket_Priority"]).copy()
    if labeled.empty:
        print("No labeled Ticket_Priority rows available to train classifier.")
        sys.exit(1)

    X = labeled["clean_text"].astype(str)
    y = labeled["Ticket_Priority"].astype(str)

    # Build and train a TF-IDF + LogisticRegression pipeline
    clf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    clf_pipeline.fit(X, y)

    # Save the trained priority model pipeline
    joblib.dump(clf_pipeline, "priority_model.pkl")
    print("Saved priority model to priority_model.pkl")

    # Apply combined scoring to all rows
    df["predicted_priority"] = df["clean_text"].apply(lambda t: score_priority(t, clf_pipeline))

    # Evaluate against ground-truth where available
    eval_df = df.dropna(subset=["Ticket_Priority"]).copy()
    if not eval_df.empty:
        y_true = eval_df["Ticket_Priority"].astype(str)
        y_pred = eval_df["predicted_priority"].astype(str)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))

        print(f"Accuracy (predicted vs. Ticket_Priority): {acc:.4f}")
        print("Confusion Matrix rows=actual, cols=predicted")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
    else:
        print("No ground-truth Ticket_Priority values available for evaluation.")

    # Persist predictions
    df.to_csv("data/priority_predictions.csv", index=False)
    print("Saved predictions to data/priority_predictions.csv")


if __name__ == "__main__":
    main()
