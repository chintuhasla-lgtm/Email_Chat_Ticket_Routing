"""03_train_department_classifier.py
Train a department (Ticket Type) classifier from cleaned ticket text.

Steps:
- loads `data/cleaned_tickets.csv`
- splits into train/test (80/20) stratified on `Ticket_Type`
- builds a Pipeline(TfidfVectorizer, LinearSVC) and fits it
- evaluates accuracy, macro-F1, and prints a classification report
- saves the fitted pipeline to `department_model.pkl`
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import sys


def main():
    # Load cleaned dataset
    try:
        df = pd.read_csv("data/cleaned_tickets.csv")
    except FileNotFoundError:
        print("Error: data/cleaned_tickets.csv not found. Run 02_text_cleaning.py first.")
        sys.exit(1)

    # Drop rows without text or ticket type
    df = df.dropna(subset=["clean_text", "Ticket_Type"]).reset_index(drop=True)

    X = df["clean_text"].astype(str)
    y = df["Ticket_Type"].astype(str)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline with TF-IDF and LinearSVC
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            ("clf", LinearSVC()),
        ]
    )

    # Fit the model
    pipe.fit(X_train, y_train)

    # Evaluate on test set
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4))

    # Save the trained pipeline
    joblib.dump(pipe, "department_model.pkl")
    print("Saved trained pipeline to department_model.pkl")


if __name__ == "__main__":
    main()
