"""
Group 23 - Community Complaint Categorization
Part 4: Recreate the process in Python.

Primary algorithm : TF-IDF + Logistic Regression
Comparison        : TF-IDF + Multinomial Naive Bayes

Run:  python train_model.py
Needs: test_records.csv in the same folder.
"""

import re
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up window needed
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

CSV_PATH = "test_records.csv"
RANDOM_STATE = 42

# Category -> office (the "routing suggestion")
ROUTING = {
    "Waste and sanitation": "Sanitation Office",
    "Road and sidewalk repair": "Public Works Office",
    "Drainage and flooding": "Drainage Services",
    "Water supply and leaks": "Water Utility",
    "Street lighting": "Electrical Maintenance",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, collapse extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize categories (strip spaces, fix capitalization)
    df["complaint_category"] = df["complaint_category"].astype(str).str.strip()
    df["complaint_category"] = df["complaint_category"].str.capitalize()

    # Check blank descriptions, then drop them
    blanks = df["complaint_text"].isna() | (df["complaint_text"].astype(str).str.strip() == "")
    print(f"Blank descriptions found: {blanks.sum()}")
    df = df[~blanks]

    # Remove duplicates (same complaint text)
    before = len(df)
    df = df.drop_duplicates(subset="complaint_text")
    print(f"Duplicates removed: {before - len(df)}")

    # Clean the text. Keep complaint_id untouched (reference only).
    df["clean_text"] = df["complaint_text"].apply(clean_text)
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.Series:
    """
    X = the cleaned complaint text ONLY.
    Location, channel and date are collected by the app but NOT fed to the model:
    in this dataset they follow a repeating pattern unrelated to the category, so
    they only add noise (and could cause unfair routing). complaint_id is never used.
    """
    return df["clean_text"]


def make_pipeline(model_name: str) -> Pipeline:
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words="english", sublinear_tf=True)
    if model_name == "Logistic Regression":
        clf = LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = MultinomialNB(alpha=0.1)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    df = load_and_prepare(CSV_PATH)
    print(f"Records after cleaning: {len(df)}")
    print(df["complaint_category"].value_counts(), "\n")

    X = build_features(df)
    y = df["complaint_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name in ["Logistic Regression", "Naive Bayes"]:
        pipe = make_pipeline(name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        pipelines[name] = (pipe, pred)

        cv_acc = cross_val_score(make_pipeline(name), X, y, cv=cv, scoring="accuracy").mean()
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0),
            "5-fold CV accuracy": cv_acc,
        })

        print(f"===== {name} =====")
        print(classification_report(y_test, pred, zero_division=0))

        # Confusion matrix picture
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=30, colorbar=False)
        ax.set_title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"confusion_{name.replace(' ', '_').lower()}.png", dpi=120)
        plt.close()

    # Comparison table + chart
    comp = pd.DataFrame(results).round(3)
    comp.to_csv("model_comparison.csv", index=False)
    print("\nMODEL COMPARISON\n", comp.to_string(index=False))

    comp.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0, 1.05), rot=0)
    plt.title("Logistic Regression vs Naive Bayes")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.close()

    # Save the primary (selected) pipeline, trained on ALL data
    final = make_pipeline("Logistic Regression")
    final.fit(X, y)
    joblib.dump({"pipeline": final, "routing": ROUTING}, "complaint_pipeline.joblib")
    print("\nSaved: complaint_pipeline.joblib, model_comparison.csv/.png, confusion_*.png")

    # Quick edge-case tests (Part 6)
    tests = {
        "Normal":      "The streetlight on my corner is broken and it is dark at night.",
        "Boundary":    "Leak",
        "Uncertain":   "Water and garbage are all over the road after the flood.",
        "Unrelated":   "I would like to ask about my business permit.",
    }
    print("\nEDGE-CASE TESTS")
    for label, text in tests.items():
        feat = clean_text(text)
        proba = final.predict_proba([feat])[0]
        top = final.classes_[proba.argmax()]
        print(f"[{label}] '{text}' -> {top} ({proba.max():.0%}) -> {ROUTING[top]}")


if __name__ == "__main__":
    main()
