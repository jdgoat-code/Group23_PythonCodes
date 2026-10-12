"""Focused checks for the revised complaint workflow."""

import unittest
from datetime import date

import joblib
import pandas as pd

from complaint_logic import resolution_date, route_for
from train_model import BASE_DIR


class FeatureChangeTests(unittest.TestCase):
    def test_branch_queue_and_office(self):
        self.assertEqual(
            route_for("Road and sidewalk repair", "E", "High"),
            "Branch 1 Public Works Office - Rapid Response",
        )
        self.assertEqual(
            route_for("Street lighting", "V", "Low"),
            "Branch 2 Electrical Maintenance - General Queue",
        )

    def test_sla_offsets(self):
        submitted = date(2026, 10, 9)
        self.assertEqual(resolution_date(submitted, "High"), date(2026, 10, 10))
        self.assertEqual(resolution_date(submitted, "Medium"), date(2026, 10, 12))
        self.assertEqual(resolution_date(submitted, "Low"), date(2026, 10, 16))

    def test_batch_and_invalid_row(self):
        from app import process_batch

        pipeline = joblib.load(BASE_DIR / "complaint_pipeline.joblib")["pipeline"]
        records = pd.read_csv(BASE_DIR / "test_records.csv", dtype=str)
        result = process_batch(pipeline, records)
        self.assertEqual(len(result), 100)
        self.assertTrue((result["estimated_resolution_date"] ==
                         records["estimated_resolution_date"]).all())
        bad = records.head(1).copy()
        bad.loc[bad.index[0], "location"] = "Q"
        with self.assertRaisesRegex(ValueError, "CSV row 2.*location"):
            process_batch(pipeline, bad)


if __name__ == "__main__":
    unittest.main()
