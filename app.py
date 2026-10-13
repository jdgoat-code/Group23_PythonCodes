"""Streamlit interface for the revised community complaint schema."""

from datetime import date, datetime
from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st

from complaint_logic import (
    CHANNELS, LOCATIONS, URGENCY_LEVELS, clean_text, resolution_date,
    route_for, validate_id, validate_record,
)

MODEL_PATH = Path(__file__).resolve().parent / "complaint_pipeline.joblib"
CONFIDENCE_THRESHOLD = 0.30
MARGIN_THRESHOLD = 0.10
INPUT_COLUMNS = ("complaint_id", "complaint_text", "urgency_level", "location",
                 "submission_channel", "date")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("schema_version") != 2:
        raise ValueError("The saved model uses the old schema. Run `python train_model.py`.")
    return bundle["pipeline"]


def predict(pipeline, records: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    features = pd.DataFrame({
        "clean_text": records["complaint_text"].map(clean_text),
        "urgency_level": records["urgency_level"],
    })
    probabilities = pipeline.predict_proba(features)
    order = probabilities.argsort(axis=1)[:, ::-1]
    top = order[:, 0]
    runner_up = order[:, 1]
    result = records.copy()
    result["complaint_category"] = pipeline.classes_[top]
    result["confidence"] = probabilities.max(axis=1).round(3)
    result["needs_review"] = ((probabilities.max(axis=1) < CONFIDENCE_THRESHOLD) |
                              (probabilities[range(len(records)), top] -
                               probabilities[range(len(records)), runner_up] < MARGIN_THRESHOLD))
    result["routing_suggestion"] = [
        "Manual review" if review else route_for(category, location, urgency)
        for category, location, urgency, review in zip(
            result["complaint_category"], result["location"],
            result["urgency_level"], result["needs_review"])
    ]
    result["estimated_resolution_date"] = [
        resolution_date(submitted, urgency).isoformat()
        for submitted, urgency in zip(result["date"], result["urgency_level"])
    ]
    return result, probabilities


def parse_date(value) -> date | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def process_batch(pipeline, uploaded: pd.DataFrame) -> pd.DataFrame:
    missing = set(INPUT_COLUMNS) - set(uploaded.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if uploaded.empty:
        raise ValueError("The CSV has no records.")
    records = uploaded.loc[:, INPUT_COLUMNS].copy().fillna("")
    records["date"] = records["date"].map(parse_date)
    errors = []
    for row_number, row in enumerate(records.itertuples(index=False), start=2):
        record_id, text, urgency, location, channel, submitted = row
        issues = validate_record(text, urgency, location, channel, submitted)
        id_issue = validate_id(record_id)
        if id_issue:
            issues.append(id_issue)
        if issues:
            errors.append(f"CSV row {row_number}: {' '.join(issues)}")
    if errors:
        raise ValueError("\n".join(errors[:10]) +
                         (f"\n...and {len(errors) - 10} more invalid rows." if len(errors) > 10 else ""))
    if records["complaint_id"].duplicated().any():
        raise ValueError("Complaint IDs must be unique within the CSV.")
    result, _ = predict(pipeline, records)
    result["date"] = result["date"].map(date.isoformat)
    return result


st.set_page_config(page_title="Complaint Categorizer", page_icon="📋")
st.title("📋 Community Complaint Categorization")
st.write("Enter a complaint to estimate its category, office queue, and resolution date.")

try:
    pipeline = load_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"{exc} Run `python train_model.py` to create the revised model.")
    st.stop()

tab_single, tab_batch, tab_analytics = st.tabs(
    ["Single complaint", "Upload CSV", "Channel analytics"]
)

with tab_single:
    with st.form("complaint_form"):
        text = st.text_area("Complaint text", height=120,
                            placeholder="e.g. The streetlamp near the school does not turn on.")
        urgency = st.selectbox("Urgency level", URGENCY_LEVELS)
        location = st.selectbox("Location", LOCATIONS)
        channel = st.selectbox("Submission channel", CHANNELS)
        submitted = st.date_input("Submission date", value=date.today())
        go = st.form_submit_button("Categorize")

    if go:
        problems = validate_record(text, urgency, location, channel, submitted)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            record = pd.DataFrame([{
                "complaint_text": text,
                "urgency_level": urgency,
                "location": location,
                "submission_channel": channel,
                "date": submitted,
            }])
            result, probabilities = predict(pipeline, record)
            row = result.iloc[0]
            if row["needs_review"]:
                st.warning(f"Uncertain category: **{row['complaint_category']}** "
                           f"({row['confidence']:.0%}). Please review before routing.")
            else:
                st.success("Prediction ready.")
            st.markdown(f"**Complaint category:** {row['complaint_category']}  \n"
                        f"**Routing suggestion:** {row['routing_suggestion']}  \n"
                        f"**Estimated resolution date:** {row['estimated_resolution_date']}  \n"
                        f"**Confidence:** {row['confidence']:.0%}")
            st.caption("Category uses complaint text and urgency. Location selects the branch; "
                       "urgency selects the queue and resolution interval. Channel is recorded "
                       "for intake analytics.")
            st.bar_chart(pd.Series(probabilities[0], index=pipeline.classes_, name="Probability"))

with tab_batch:
    st.write("Upload a CSV with `complaint_id`, `complaint_text`, `urgency_level`, "
             "`location`, `submission_channel`, and `date` (YYYY-MM-DD).")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        try:
            uploaded = pd.read_csv(file, dtype=str)
            results = process_batch(pipeline, uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.dataframe(results)
            st.download_button("Download results", results.to_csv(index=False),
                               "predictions.csv", mime="text/csv")
            st.session_state["channel_counts"] = results["submission_channel"].value_counts()

with tab_analytics:
    st.write("Submission channels in the latest valid CSV upload.")
    if "channel_counts" in st.session_state:
        st.bar_chart(st.session_state["channel_counts"])
    else:
        st.info("Upload a CSV to see channel utilization.")
