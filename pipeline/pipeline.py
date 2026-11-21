#!/usr/bin/env python3
"""Build the Shopping Sessions daily tables with DuckDB."""

import csv
import glob
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

import duckdb

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))

DAY_MS = 86_400_000
SESSION_GAP_MS = 1_800_000
CLICK_WINDOW_MS = 604_800_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
F1_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z"
)


def parse_csv_line(line):
    """Parse the restricted CSV dialect; return None if malformed."""
    if not line:
        return None

    fields = []
    length = len(line)
    position = 0
    while True:
        if position < length and line[position] == '"':
            end = line.find('"', position + 1)
            if end < 0:
                return None
            value = line[position + 1:end]
            if "," in value:
                return None
            position = end + 1
            if position < length and line[position] != ",":
                return None
        else:
            end = line.find(",", position)
            if end < 0:
                end = length
            value = line[position:end]
            if '"' in value:
                return None
            position = end

        fields.append(value if value else None)
        if position == length:
            break
        position += 1
        if position == length:
            fields.append(None)
            break
    return fields


def iter_csv_rows(path, expected_columns):
    """Yield well-formed data rows, treating only LF as a line ending."""
    with open(path, "r", encoding="utf-8", newline="\n") as source:
        next(source, None)
        for raw_line in source:
            line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            fields = parse_csv_line(line)
            if fields is not None and len(fields) == expected_columns:
                yield fields


def manifest_timestamp_ms(value):
    match = F1_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid manifest timestamp: {value!r}")
    timestamp = datetime(*(int(part) for part in match.groups()), tzinfo=timezone.utc)
    delta = timestamp - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000


class UserIds:
    """Union-find that keeps each person's byte-wise smallest ID as its root."""

    def __init__(self):
        self.parent = {}

    def find(self, user_id):
        if user_id not in self.parent:
            self.parent[user_id] = user_id
            return user_id
        root = user_id
        while self.parent[root] != root:
            root = self.parent[root]
        current = user_id
        while self.parent[current] != current:
            next_id = self.parent[current]
            self.parent[current] = root
            current = next_id
        return root

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if first_root.encode("utf-8") < second_root.encode("utf-8"):
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def sql_string(value):
    return "'" + value.replace("'", "''") + "'"


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerows(rows)


def normalize_files(paths, column_count, output_path, require_first=False):
    def rows():
        for path in paths:
            for row in iter_csv_rows(path, column_count):
                if not require_first or row[0] is not None:
                    yield row

    write_rows(output_path, rows())


def load_normalized(connection, table, path, columns):
    schema = ", ".join(f"{sql_string(column)}: 'VARCHAR'" for column in columns)
    connection.execute(
        f"""
        CREATE TEMP TABLE {table} AS
        SELECT * FROM read_csv(
            ?, auto_detect=false, header=false, columns={{{schema}}},
            nullstr='', allow_quoted_nulls=true, new_line='\\n'
        )
        """,
        [path],
    )


def main():
    started = time.perf_counter()
    previous = started

    def log_stage(name):
        nonlocal previous
        now = time.perf_counter()
        print(
            f"{name}: {now - previous:.3f}s (total {now - started:.3f}s)",
            file=sys.stderr,
            flush=True,
        )
        previous = now

    with open(os.path.join(DATA, "manifest.json"), "r", encoding="utf-8") as source:
        manifest = json.load(source)
    window_start_ms = manifest_timestamp_ms(manifest["window_start"])
    window_end_ms = manifest_timestamp_ms(manifest["window_end"])
    if window_start_ms >= window_end_ms:
        raise ValueError("manifest must contain a valid, increasing time window")

    connection = duckdb.connect()
    connection.execute(f"SET threads = {NCPU}")
    connection.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")
    connection.execute(f"SET temp_directory = {sql_string(os.path.join(SCRATCH, 'duckdb_tmp'))}")
    connection.execute("SET preserve_insertion_order = false")

    os.makedirs(SCRATCH, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shopping_sessions_", dir=SCRATCH) as staging:
        user_ids = UserIds()
        for path in sorted(glob.glob(os.path.join(DATA, "links", "*.csv"))):
            for first, second in iter_csv_rows(path, 2):
                if first is not None and second is not None:
                    user_ids.union(first, second)
        link_path = os.path.join(staging, "user_ids.csv")
        write_rows(link_path, ((user_id, user_ids.find(user_id)) for user_id in user_ids.parent))
        load_normalized(connection, "user_ids", link_path, ["user_id", "person_id"])
        del user_ids
        log_stage("links")

        for table, filename, columns in (
            ("products", "products.csv", ["product_id", "category"]),
            ("campaigns", "campaigns.csv", ["campaign_id", "channel"]),
        ):
            path = os.path.join(staging, table + ".csv")
            normalize_files([os.path.join(DATA, filename)], len(columns), path)
            load_normalized(connection, table, path, columns)
        log_stage("dimensions")

        connection.execute(
            """
            CREATE MACRO timestamp_ms(value) AS (
                CASE
                    WHEN regexp_full_match(value, '[0-9]+') THEN
                        try_cast(coalesce(nullif(ltrim(value, '0'), ''), '0') AS BIGINT)
                    WHEN regexp_full_match(
                        value,
                        '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z'
                    ) AND substring(value, 1, 4) <> '0000' THEN
                        epoch_ms(try_strptime(substring(value, 1, 19), '%Y-%m-%dT%H:%M:%S'))
                    WHEN regexp_full_match(
                        value,
                        '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}'
                    ) AND substring(value, 1, 4) <> '0000'
                      AND try_cast(substring(value, 21, 2) AS INTEGER) <= 23
                      AND try_cast(substring(value, 24, 2) AS INTEGER) <= 59 THEN
                        epoch_ms(try_strptime(substring(value, 1, 19), '%Y-%m-%dT%H:%M:%S'))
                        - CASE WHEN substring(value, 20, 1) = '+' THEN 1 ELSE -1 END
                          * (try_cast(substring(value, 21, 2) AS INTEGER) * 60
                             + try_cast(substring(value, 24, 2) AS INTEGER)) * 60000
                    ELSE NULL
                END
            )
            """
        )
        event_path = os.path.join(staging, "events.csv")
        normalize_files(
            sorted(glob.glob(os.path.join(DATA, "events", "*.csv"))),
            7, event_path, require_first=True,
        )
        load_normalized(
            connection, "event_raw", event_path,
            ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"],
        )
        log_stage("events parse and bulk load")

        connection.execute(
            f"""
            CREATE TEMP TABLE events AS
            WITH deduplicated AS (
                SELECT * FROM event_raw
                QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY ingest_ts) = 1
            ),
            parsed AS (
                SELECT *, timestamp_ms(ts) AS ts_ms,
                       CASE WHEN regexp_full_match(price_cents, '[0-9]+')
                            THEN try_cast(coalesce(nullif(ltrim(price_cents, '0'), ''), '0') AS BIGINT)
                       END AS price
                FROM deduplicated
                WHERE user_id IS NOT NULL AND product_id IS NOT NULL
                  AND event_type IN ('view', 'cart', 'purchase')
            )
            SELECT coalesce(ids.person_id, e.user_id) AS person_id,
                   e.ts_ms, e.event_type, e.product_id,
                   CASE WHEN e.event_type = 'purchase' THEN e.price ELSE 0 END AS price_cents
            FROM parsed e
            LEFT JOIN user_ids ids ON ids.user_id = e.user_id
            WHERE e.ts_ms >= {window_start_ms} AND e.ts_ms < {window_end_ms}
              AND epoch_ms(strptime(e.ingest_ts, '%Y-%m-%dT%H:%M:%SZ')) - e.ts_ms <= {DAY_MS}
              AND (e.event_type <> 'purchase' OR e.price BETWEEN 1 AND 10000000)
            """
        )
        connection.execute("DROP TABLE event_raw")
        os.remove(event_path)
        log_stage("events deduplication and validation")

        click_path = os.path.join(staging, "clicks.csv")
        normalize_files(
            sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv"))),
            4, click_path, require_first=True,
        )
        load_normalized(
            connection, "click_raw", click_path,
            ["click_id", "user_id", "campaign_id", "ts"],
        )
        log_stage("clicks parse and bulk load")

        connection.execute(
            f"""
            CREATE TEMP TABLE clicks AS
            WITH deduplicated AS (
                SELECT * FROM click_raw
                QUALIFY row_number() OVER (PARTITION BY click_id) = 1
            ),
            parsed AS (
                SELECT *, timestamp_ms(ts) AS ts_ms FROM deduplicated
                WHERE user_id IS NOT NULL
            )
            SELECT c.click_id, coalesce(ids.person_id, c.user_id) AS person_id,
                   c.campaign_id, c.ts_ms
            FROM parsed c
            JOIN campaigns campaigns ON campaigns.campaign_id = c.campaign_id
            LEFT JOIN user_ids ids ON ids.user_id = c.user_id
            WHERE c.ts_ms >= {window_start_ms - CLICK_WINDOW_MS}
              AND c.ts_ms < {window_end_ms}
            """
        )
        connection.execute("DROP TABLE click_raw")
        connection.execute("DROP TABLE user_ids")
        os.remove(click_path)
        log_stage("clicks deduplication and validation")

        connection.execute(
            f"""
            CREATE TEMP TABLE session_events AS
            WITH ordered AS (
                SELECT *, lag(ts_ms) OVER (PARTITION BY person_id ORDER BY ts_ms) AS previous_ts
                FROM events
            ),
            boundaries AS (
                SELECT *, CASE WHEN previous_ts IS NULL OR ts_ms - previous_ts > {SESSION_GAP_MS}
                               THEN 1 ELSE 0 END AS starts_session
                FROM ordered
            )
            SELECT person_id, ts_ms, event_type, product_id, price_cents,
                   sum(starts_session) OVER (
                       PARTITION BY person_id ORDER BY ts_ms
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS session_number
            FROM boundaries
            """
        )
        connection.execute("DROP TABLE events")
        connection.execute(
            """
            CREATE TEMP TABLE sessions AS
            SELECT person_id, session_number, min(ts_ms) AS session_start_ms,
                   cast(epoch_ms(min(ts_ms)) AS DATE) AS day,
                   count(*)::BIGINT AS event_count,
                   sum(price_cents)::BIGINT AS revenue_cents,
                   count(*) FILTER (WHERE event_type = 'purchase')::BIGINT AS purchase_count
            FROM session_events
            GROUP BY person_id, session_number
            """
        )
        log_stage("sessionization")

        connection.execute(
            """
            CREATE TEMP TABLE user_daily AS
            WITH category_totals AS (
                SELECT s.person_id, s.day, coalesce(p.category, 'unknown') AS category,
                       sum(e.price_cents) AS category_revenue
                FROM session_events e
                JOIN sessions s USING (person_id, session_number)
                LEFT JOIN products p ON p.product_id = e.product_id
                WHERE e.event_type = 'purchase'
                GROUP BY s.person_id, s.day, coalesce(p.category, 'unknown')
            ),
            category_daily AS (
                SELECT person_id, day, category
                FROM category_totals
                QUALIFY row_number() OVER (
                    PARTITION BY person_id, day ORDER BY category_revenue DESC, category ASC
                ) = 1
            ),
            daily AS (
                SELECT person_id, day, count(*)::BIGINT AS sessions,
                       sum(event_count)::BIGINT AS events,
                       sum(revenue_cents)::BIGINT AS revenue_cents
                FROM sessions GROUP BY person_id, day
            )
            SELECT d.person_id AS user_id, d.day, d.sessions, d.events, d.revenue_cents,
                   c.category AS top_category
            FROM daily d LEFT JOIN category_daily c USING (person_id, day)
            """
        )
        connection.execute("DROP TABLE session_events")
        connection.execute("DROP TABLE products")
        log_stage("user daily aggregation")

        connection.execute(
            f"""
            CREATE TEMP TABLE session_campaign AS
            WITH click_at_time AS (
                SELECT person_id, ts_ms, arg_min(campaign_id, click_id) AS campaign_id
                FROM clicks GROUP BY person_id, ts_ms
            )
            SELECT s.person_id, s.session_number, s.day, s.purchase_count, s.revenue_cents,
                   CASE WHEN c.ts_ms >= s.session_start_ms - {CLICK_WINDOW_MS}
                        THEN c.campaign_id ELSE 'organic' END AS campaign_id
            FROM sessions s ASOF LEFT JOIN click_at_time c
              ON s.person_id = c.person_id AND s.session_start_ms >= c.ts_ms
            """
        )
        log_stage("last-click attribution")

        # Merge overlapping click-eligibility intervals, then count sessions in
        # each interval via two prefix lookups instead of a session-click join.
        connection.execute(
            f"""
            CREATE TEMP TABLE assisted_daily AS
            WITH click_times AS (
                SELECT DISTINCT person_id, campaign_id, ts_ms FROM clicks
            ),
            previous_click AS (
                SELECT *, lag(ts_ms) OVER (
                    PARTITION BY person_id, campaign_id ORDER BY ts_ms
                ) AS previous_ts
                FROM click_times
            ),
            islands AS (
                SELECT *, sum(
                    CASE WHEN previous_ts IS NULL OR ts_ms - previous_ts > {CLICK_WINDOW_MS}
                         THEN 1 ELSE 0 END
                ) OVER (
                    PARTITION BY person_id, campaign_id ORDER BY ts_ms
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS island
                FROM previous_click
            ),
            intervals AS (
                SELECT person_id, campaign_id, min(ts_ms) AS start_ms,
                       max(ts_ms) + {CLICK_WINDOW_MS} AS end_ms
                FROM islands GROUP BY person_id, campaign_id, island
            ),
            days AS (SELECT DISTINCT day FROM sessions),
            daily_intervals AS (
                SELECT i.person_id, i.campaign_id, d.day,
                       greatest(i.start_ms, epoch_ms(d.day::TIMESTAMP)) AS start_ms,
                       least(i.end_ms, epoch_ms(d.day::TIMESTAMP) + {DAY_MS} - 1) AS end_ms
                FROM intervals i JOIN days d
                  ON d.day BETWEEN cast(epoch_ms(i.start_ms) AS DATE)
                               AND cast(epoch_ms(i.end_ms) AS DATE)
            ),
            session_points AS (
                SELECT person_id, session_start_ms,
                       row_number() OVER (PARTITION BY person_id ORDER BY session_start_ms) AS n
                FROM sessions
            ),
            end_counts AS (
                SELECT i.*, coalesce(s.n, 0) AS end_count
                FROM daily_intervals i ASOF LEFT JOIN session_points s
                  ON i.person_id = s.person_id AND i.end_ms >= s.session_start_ms
            ),
            interval_counts AS (
                SELECT i.campaign_id, i.day, i.end_count - coalesce(s.n, 0) AS count
                FROM end_counts i ASOF LEFT JOIN session_points s
                  ON i.person_id = s.person_id AND i.start_ms > s.session_start_ms
            )
            SELECT campaign_id, day, sum(count)::BIGINT AS assisted_sessions
            FROM interval_counts
            GROUP BY campaign_id, day
            HAVING sum(count) > 0
            """
        )
        connection.execute("DROP TABLE clicks")
        connection.execute("DROP TABLE sessions")
        log_stage("assist interval aggregation")

        connection.execute(
            """
            CREATE TEMP TABLE campaign_daily AS
            WITH credited AS (
                SELECT campaign_id, day, count(*)::BIGINT AS sessions,
                       count(*) FILTER (WHERE purchase_count > 0)::BIGINT AS purchasing_sessions,
                       sum(revenue_cents)::BIGINT AS revenue_cents
                FROM session_campaign GROUP BY campaign_id, day
            ),
            combined AS (
                SELECT coalesce(c.campaign_id, a.campaign_id) AS campaign_id,
                       coalesce(c.day, a.day) AS day,
                       coalesce(c.sessions, 0)::BIGINT AS sessions,
                       coalesce(c.purchasing_sessions, 0)::BIGINT AS purchasing_sessions,
                       coalesce(c.revenue_cents, 0)::BIGINT AS revenue_cents,
                       coalesce(a.assisted_sessions, 0)::BIGINT AS assisted_sessions
                FROM credited c FULL OUTER JOIN assisted_daily a
                  ON a.campaign_id = c.campaign_id AND a.day = c.day
            )
            SELECT c.campaign_id,
                   CASE WHEN c.campaign_id = 'organic' THEN 'organic' ELSE campaigns.channel END AS channel,
                   c.day, c.sessions, c.purchasing_sessions, c.revenue_cents, c.assisted_sessions
            FROM combined c LEFT JOIN campaigns USING (campaign_id)
            """
        )
        connection.execute("DROP TABLE session_campaign")
        connection.execute("DROP TABLE assisted_daily")
        log_stage("campaign daily aggregation")

        for table in ("user_daily", "campaign_daily"):
            output_dir = os.path.join(OUT, table)
            os.makedirs(output_dir, exist_ok=True)
            connection.execute(
                f"COPY (SELECT * FROM {table}) TO "
                f"{sql_string(os.path.join(output_dir, 'part-0.parquet'))} (FORMAT PARQUET)"
            )
        log_stage("write parquet")


if __name__ == "__main__":
    main()
