"""Regression tests for bulk loading and dense click attribution."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
WEEK_MS = 604_800_000
BASE_MS = 1_790_812_800_000  # 2026-10-01 00:00:00 UTC


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class PipelineRegressionTest(unittest.TestCase):
    def test_strict_csv_long_epoch_and_disjoint_assists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output, scratch = root / "data", root / "output", root / "scratch"
            for name in ("events", "clicks", "links"):
                (data / name).mkdir(parents=True)
            scratch.mkdir()
            (data / "manifest.json").write_text(
                '{"window_start":"2026-10-01T00:00:00Z",'
                '"window_end":"2026-10-15T00:00:00Z"}',
                encoding="utf-8",
            )
            write_csv(data / "links" / "part.csv", ["user_id_a", "user_id_b"], [])
            write_csv(data / "products.csv", ["product_id", "category"], [("p1", "books")])
            write_csv(data / "campaigns.csv", ["campaign_id", "channel"], [("c1", "search")])
            events_path = data / "events" / "part.csv"
            times = [BASE_MS, BASE_MS + WEEK_MS, BASE_MS + WEEK_MS + 1_860_000]
            write_csv(
                events_path,
                ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"],
                [(f"e{i}", "u1", "0" * 5000 + str(timestamp),
                  "2026-09-01T00:00:00Z", "view", "p1", "")
                 for i, timestamp in enumerate(times)],
            )
            with events_path.open("a", encoding="utf-8", newline="") as output_file:
                output_file.write(
                    '"bad,id",u1,1790812800000,2026-09-01T00:00:00Z,view,p1,\n'
                )
            write_csv(
                data / "clicks" / "part.csv", ["click_id", "user_id", "campaign_id", "ts"],
                [("a", "u1", "c1", BASE_MS), ("b", "u1", "c1", BASE_MS + WEEK_MS + 1)],
            )
            env = dict(os.environ, DATA_DIR=str(data), OUTPUT_DIR=str(output),
                       SCRATCH_DIR=str(scratch), NCPU="4", MEM_GB="2")
            subprocess.run([sys.executable, str(ROOT / "pipeline" / "pipeline.py")],
                           env=env, check=True, capture_output=True, text=True)
            db = duckdb.connect()
            users = db.execute(
                "SELECT sum(sessions), sum(events) FROM read_parquet(?)",
                [str(output / "user_daily" / "*.parquet")],
            ).fetchone()
            campaigns = db.execute(
                "SELECT sum(sessions), sum(assisted_sessions) FROM read_parquet(?)",
                [str(output / "campaign_daily" / "*.parquet")],
            ).fetchone()
            self.assertEqual(users, (3, 3))
            self.assertEqual(campaigns, (3, 3))

    def test_bulk_load_and_dense_assist_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output, scratch = root / "data", root / "output", root / "scratch"
            for name in ("events", "clicks", "links"):
                (data / name).mkdir(parents=True)
            scratch.mkdir()
            (data / "manifest.json").write_text(
                '{"window_start":"2026-10-01T00:00:00Z",'
                '"window_end":"2026-10-15T00:00:00Z"}',
                encoding="utf-8",
            )
            write_csv(data / "links" / "part.csv", ["user_id_a", "user_id_b"], [("u2", "u1")])
            write_csv(data / "products.csv", ["product_id", "category"], [("p1", "books")])
            write_csv(
                data / "campaigns.csv", ["campaign_id", "channel"],
                [("c1", "search"), ("c2", "social"), ("c3", "email")],
            )

            session_times = [BASE_MS + i * 1_860_000 for i in range(250)]
            events = [
                (f"e{i}", "u2", timestamp, "2026-09-01T00:00:00Z",
                 "purchase" if i % 7 == 0 else "view", "p1",
                 "10" if i % 7 == 0 else "")
                for i, timestamp in enumerate(session_times)
            ]
            events.extend(
                (f"extra{i}", "u1", session_times[100], "2026-09-01T00:00:00Z", "view", "p1", "")
                for i in range(10_020)
            )
            write_csv(
                data / "events" / "part.csv",
                ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"],
                events,
            )
            clicks = [
                (f"k{i:04}", "u1", f"c{i % 3 + 1}", BASE_MS - 2 * WEEK_MS + i * 1_200_000)
                for i in range(1300)
                if BASE_MS - 2 * WEEK_MS + i * 1_200_000 >= BASE_MS - WEEK_MS
            ]
            clicks.extend([
                ("a", "u2", "c2", session_times[100]),
                ("z", "u2", "c3", session_times[100]),
                ("boundary", "u2", "c3", session_times[0] - WEEK_MS),
            ])
            write_csv(data / "clicks" / "part.csv", ["click_id", "user_id", "campaign_id", "ts"], clicks)

            env = dict(os.environ, DATA_DIR=str(data), OUTPUT_DIR=str(output),
                       SCRATCH_DIR=str(scratch), NCPU="4", MEM_GB="2")
            subprocess.run([sys.executable, str(ROOT / "pipeline" / "pipeline.py")],
                           env=env, check=True, capture_output=True, text=True)
            db = duckdb.connect()
            users = db.execute("SELECT * FROM read_parquet(?)", [str(output / "user_daily" / "*.parquet")]).fetchall()
            campaigns = db.execute("SELECT * FROM read_parquet(?)", [str(output / "campaign_daily" / "*.parquet")]).fetchall()

            expected_users = defaultdict(lambda: [0, 0, 0])
            expected_campaigns = defaultdict(lambda: [0, 0, 0, 0])
            for i, timestamp in enumerate(session_times):
                day = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp)).date()
                purchase = i % 7 == 0
                user = expected_users[day]
                user[0] += 1
                user[1] += 1 + (10_020 if i == 100 else 0)
                user[2] += 10 if purchase else 0
                eligible = [click for click in clicks if timestamp - WEEK_MS <= click[3] <= timestamp]
                if eligible:
                    winner = min(eligible, key=lambda click: (-click[3], click[0].encode("utf-8")))
                    credited = winner[2]
                else:
                    credited = "organic"
                counts = expected_campaigns[(credited, day)]
                counts[0] += 1
                counts[1] += purchase
                counts[2] += 10 if purchase else 0
                for campaign in {click[2] for click in eligible}:
                    expected_campaigns[(campaign, day)][3] += 1

            self.assertEqual(
                Counter(users),
                Counter(("u1", day, *values, "books" if values[2] else None)
                        for day, values in expected_users.items()),
            )
            channels = {"c1": "search", "c2": "social", "c3": "email", "organic": "organic"}
            self.assertEqual(
                Counter(campaigns),
                Counter((campaign, channels[campaign], day, *values)
                        for (campaign, day), values in expected_campaigns.items()),
            )


if __name__ == "__main__":
    unittest.main()
