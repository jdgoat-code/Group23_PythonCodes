"""
Group 23 - Community Complaint Categorization
Part 5: Streamlit interface.

Run:  streamlit run app.py
Needs: complaint_pipeline.joblib (created by train_model.py)
"""

import re
from datetime import date

import joblib
import pandas as pd
import streamlit as st

CONFIDENCE_THRESHOLD = 0.30   # below this -> "uncertain, review manually"
MIN_WORDS = 3
CHANNELS = ["Web form", "Mobile app", "Phone", "Walk-in", "Email"]
LOCATIONS = [f"Zone {c}" for c in "ABCDEFGHIJ"]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    bundle = joblib.load("complaint_pipeline.joblib")
    return bundle["pipeline"], bundle["routing"]


def validate(text, location, channel, submitted):
    """Return a list of problems (empty list = input is OK)."""
    errors = []
    if not text or not text.strip():
        errors.append("Complaint text is empty.")
    elif len(clean_text(text).split()) < MIN_WORDS:
        errors.append(f"Complaint text is too short (at least {MIN_WORDS} words).")
    if not location or not location.strip():
        errors.append("Location is missing.")
    if channel not in CHANNELS:
        errors.append("Please choose a valid submission channel.")
    if submitted is None:
        errors.append("Date is missing.")
    elif submitted > date.today():
        errors.append("Date cannot be in the future.")
    return errors


st.set_page_config(page_title="Complaint Categorizer", page_icon="📋")
st.title("📋 Community Complaint Categorization")
st.write("Enter a complaint and the system will suggest a category and the office to route it to.")

try:
    pipeline, routing = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single complaint", "Upload CSV"])

# ---------------- Single complaint ----------------
with tab_single:
    with st.form("complaint_form"):
        text = st.text_area("Complaint text", height=120,
                            placeholder="e.g. The streetlamp near the school does not turn on.")
        location = st.selectbox("Location", LOCATIONS)
        channel = st.selectbox("Submission channel", CHANNELS)
        submitted = st.date_input("Submission date", value=date.today())
        go = st.form_submit_button("Categorize")

    if go:
        problems = validate(text, location, channel, submitted)
        if problems:
            for p in problems:
                st.error(p)
        else:
            feat = clean_text(text)   # model uses complaint text only
            proba = pipeline.predict_proba([feat])[0]
            order = proba.argsort()[::-1]
            top, conf = pipeline.classes_[order[0]], proba[order[0]]
            second, conf2 = pipeline.classes_[order[1]], proba[order[1]]

            if conf < CONFIDENCE_THRESHOLD or (conf - conf2) < 0.10:
                st.warning(
                    f"⚠️ Uncertain result. Best guess: **{top}** ({conf:.0%}), "
                    f"next: {second} ({conf2:.0%}). Please review manually before routing."
                )
            else:
                st.success(f"**Category:** {top}  \n**Route to:** {routing[top]}  \n"
                           f"**Confidence:** {conf:.0%}")

            st.write("Explanation: the category comes mainly from key words in the complaint text "
                     "(for example 'streetlight', 'drain', 'pothole'). Location, channel and date are recorded but not used by the model.")
            st.bar_chart(pd.Series(proba, index=pipeline.classes_, name="Probability"))

# ---------------- Batch upload ----------------
with tab_batch:
    st.write("Upload a CSV with columns: `complaint_text`, `location`, `submission_channel`, `submission_date`.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        needed = {"complaint_text", "location", "submission_channel"}
        if not needed.issubset(df.columns):
            st.error(f"Missing columns: {', '.join(needed - set(df.columns))}")
        else:
            df = df.fillna("")
            feats = df["complaint_text"].apply(clean_text)
            proba = pipeline.predict_proba(feats)
            df["predicted_category"] = pipeline.classes_[proba.argmax(axis=1)]
            df["confidence"] = proba.max(axis=1).round(2)
            df["routing_suggestion"] = df["predicted_category"].map(routing)
            df.loc[df["complaint_text"].str.strip() == "", ["predicted_category", "routing_suggestion"]] = "Needs review"
            df.loc[df["confidence"] < CONFIDENCE_THRESHOLD, "routing_suggestion"] = "Manual review"
            st.dataframe(df)
            st.download_button("Download results", df.to_csv(index=False), "predictions.csv")
