"""Train complaint category classifiers on the revised complaint schema."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from complaint_logic import CHANNELS, LOCATIONS, OFFICES, URGENCY_LEVELS, clean_text

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "test_records.csv"
RANDOM_STATE = 42


def load_and_prepare(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"complaint_id", "complaint_text", "urgency_level", "location",
                "submission_channel", "date", "complaint_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {', '.join(sorted(missing))}")

    df["complaint_category"] = df["complaint_category"].astype(str).str.strip()
    df["urgency_level"] = df["urgency_level"].astype(str).str.strip().str.title()
    blanks = df["complaint_text"].isna() | df["complaint_text"].astype(str).str.strip().eq("")
    print(f"Blank descriptions found: {blanks.sum()}")
    df = df.loc[~blanks].copy()
    before = len(df)
    df = df.drop_duplicates(subset="complaint_text").copy()
    print(f"Duplicates removed: {before - len(df)}")
    df["clean_text"] = df["complaint_text"].map(clean_text)
    df = df.loc[df["clean_text"].ne("")].copy()

    for column, allowed in (("complaint_category", OFFICES),
                            ("urgency_level", URGENCY_LEVELS),
                            ("location", LOCATIONS),
                            ("submission_channel", CHANNELS)):
        invalid = sorted(set(df[column].dropna()) - set(allowed))
        if df[column].isna().any() or invalid:
            raise ValueError(f"Invalid {column} values: {invalid}")
    if pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("Training data contains invalid dates")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Text drives category prediction; urgency is the secondary signal."""
    return df[["clean_text", "urgency_level"]]


def make_pipeline(model_name: str) -> Pipeline:
    features = ColumnTransformer([
        ("text", TfidfVectorizer(stop_words="english", sublinear_tf=True), "clean_text"),
        ("urgency", OneHotEncoder(handle_unknown="ignore"), ["urgency_level"]),
    ])
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(C=10, max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "Naive Bayes":
        classifier = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("features", features), ("clf", classifier)])


def main() -> None:
    df = load_and_prepare()
    print(f"Records after cleaning: {len(df)}")
    print(df["complaint_category"].value_counts(), "\n")
    X, y = build_features(df), df["complaint_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name in ("Logistic Regression", "Naive Bayes"):
        pipe = make_pipeline(name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
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

        labels = sorted(y.unique())
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=labels),
                               display_labels=labels).plot(ax=ax, xticks_rotation=30,
                                                           colorbar=False)
        ax.set_title(f"Confusion matrix - {name}")
        fig.tight_layout()
        fig.savefig(BASE_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=120)
        plt.close(fig)

    comp = pd.DataFrame(results).round(3)
    comp.to_csv(BASE_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON\n", comp.to_string(index=False))
    ax = comp.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0, 1.05), rot=0
    )
    ax.set_title("Logistic Regression vs Naive Bayes")
    ax.set_ylabel("Score")
    ax.figure.tight_layout()
    ax.figure.savefig(BASE_DIR / "model_comparison.png", dpi=120)
    plt.close(ax.figure)

    final = make_pipeline("Logistic Regression")
    final.fit(X, y)
    joblib.dump({"pipeline": final, "schema_version": 2}, BASE_DIR / "complaint_pipeline.joblib")
    print("\nSaved model and comparison charts")


if __name__ == "__main__":
    main()
