"""Shared schema and deterministic complaint handling rules."""

import re
from datetime import date, timedelta

CHANNELS = ("Mobile App", "Web Portal", "Hotline", "Walk-in")
LOCATIONS = tuple("ABCDEVWXYZ")
URGENCY_LEVELS = ("High", "Medium", "Low")
SLA_DAYS = {"High": 1, "Medium": 3, "Low": 7}
OFFICES = {
    "Waste and sanitation": "Sanitation Office",
    "Road and sidewalk repair": "Public Works Office",
    "Drainage and flooding": "Drainage Services",
    "Water supply and leaks": "Water Utility",
    "Street lighting": "Electrical Maintenance",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def route_for(category: str, location: str, urgency: str) -> str:
    if category not in OFFICES:
        raise ValueError(f"Unknown category: {category}")
    if location not in LOCATIONS:
        raise ValueError(f"Unknown location: {location}")
    if urgency not in URGENCY_LEVELS:
        raise ValueError(f"Unknown urgency level: {urgency}")
    branch = "Branch 1" if location in "ABCDE" else "Branch 2"
    queue = "Rapid Response" if urgency == "High" else "General Queue"
    return f"{branch} {OFFICES[category]} - {queue}"


def resolution_date(submitted: date, urgency: str) -> date:
    if urgency not in URGENCY_LEVELS:
        raise ValueError(f"Unknown urgency level: {urgency}")
    return submitted + timedelta(days=SLA_DAYS[urgency])


def validate_record(text, urgency, location, channel, submitted) -> list[str]:
    errors = []
    if not isinstance(text, str) or not text.strip():
        errors.append("Complaint text is empty.")
    elif len(clean_text(text).split()) < 3:
        errors.append("Complaint text is too short (at least 3 words).")
    if urgency not in URGENCY_LEVELS:
        errors.append("Please choose a valid urgency level.")
    if location not in LOCATIONS:
        errors.append("Please choose a valid location.")
    if channel not in CHANNELS:
        errors.append("Please choose a valid submission channel.")
    if not isinstance(submitted, date):
        errors.append("Date must be a valid YYYY-MM-DD calendar date.")
    elif submitted > date.today():
        errors.append("Date cannot be in the future.")
    return errors


def validate_id(complaint_id) -> str | None:
    if not isinstance(complaint_id, str) or not re.fullmatch(r"CC-\d{3,}", complaint_id.strip()):
        return "Complaint ID must use the format CC-###."
    return None
