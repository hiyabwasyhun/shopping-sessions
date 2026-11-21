#!/usr/bin/env python3
"""Build the Shopping Sessions daily tables with DuckDB."""

import glob
import json
import os
import re
from datetime import datetime, timedelta, timezone

import duckdb

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))

EVENT_COLUMNS = 7
CLICK_COLUMNS = 4
LINK_COLUMNS = 2
DAY_MS = 86_400_000
SESSION_GAP_MS = 1_800_000
CLICK_WINDOW_MS = 604_800_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

F1_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z"
)
F2_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})([+-])([0-9]{2}):([0-9]{2})"
)
DIGITS_PATTERN = re.compile(r"[0-9]+")


def parse_csv_line(line):
    """Parse the challenge's restricted CSV dialect; return None if malformed."""
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
    """Yield well-formed data rows, preserving carriage returns as field data."""
    with open(path, "r", encoding="utf-8", newline="\n") as source:
        next(source, None)
        for raw_line in source:
            line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            fields = parse_csv_line(line)
            if fields is not None and len(fields) == expected_columns:
                yield fields


def parse_timestamp_ms(value):
    """Parse the three allowed timestamp forms, returning None for invalid input."""
    if value is None:
        return None

    match = F1_PATTERN.fullmatch(value)
    offset_minutes = 0
    if match is None:
        match = F2_PATTERN.fullmatch(value)
        if match is None:
            if DIGITS_PATTERN.fullmatch(value) is not None:
                return int(value)
            return None
        sign = 1 if match.group(7) == "+" else -1
        offset_hours = int(match.group(8))
        offset_minute_part = int(match.group(9))
        if offset_hours > 23 or offset_minute_part > 59:
            return None
        offset_minutes = sign * (offset_hours * 60 + offset_minute_part)

    try:
        timestamp = datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None

    timestamp -= timedelta(minutes=offset_minutes)
    delta = timestamp - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000


def parse_ingest_timestamp(value):
    """Parse the always-valid F1 ingest timestamp, failing on a broken invariant."""
    match = F1_PATTERN.fullmatch(value or "")
    if match is None:
        raise ValueError(f"invalid ingest_ts (expected F1): {value!r}")
    timestamp_ms = parse_timestamp_ms(value)
    if timestamp_ms is None:
        raise ValueError(f"invalid ingest_ts (expected F1): {value!r}")
    return timestamp_ms


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

    def person_name(self, user_id):
        return self.find(user_id)


def sql_string(value):
    return "'" + value.replace("'", "''") + "'"


def insert_batches(connection, statement, rows, batch_size=10_000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            connection.executemany(statement, batch)
            batch.clear()
    if batch:
        connection.executemany(statement, batch)


def main():
    with open(os.path.join(DATA, "manifest.json"), "r", encoding="utf-8") as source:
        manifest = json.load(source)
    window_start_ms = parse_timestamp_ms(manifest["window_start"])
    window_end_ms = parse_timestamp_ms(manifest["window_end"])
    if (
        window_start_ms is None
        or window_end_ms is None
        or window_start_ms >= window_end_ms
    ):
        raise ValueError("manifest must contain a valid, increasing time window")

    connection = duckdb.connect()
    connection.execute(f"SET threads = {NCPU}")
    connection.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")
    connection.execute(f"SET temp_directory = {sql_string(os.path.join(SCRATCH, 'duckdb_tmp'))}")
    connection.execute("SET preserve_insertion_order = false")

    print("window:", manifest["window_start"], "to", manifest["window_end"])

    user_ids = UserIds()
    for path in sorted(glob.glob(os.path.join(DATA, "links", "*.csv"))):
        for first, second in iter_csv_rows(path, LINK_COLUMNS):
            if first is not None and second is not None:
                user_ids.union(first, second)

    connection.execute("CREATE TEMP TABLE products (product_id VARCHAR, category VARCHAR)")
    product_rows = []
    for product_id, category in iter_csv_rows(
        os.path.join(DATA, "products.csv"), 2
    ):
        if product_id is not None:
            product_rows.append((product_id, category))
    if product_rows:
        connection.executemany("INSERT INTO products VALUES (?, ?)", product_rows)

    connection.execute("CREATE TEMP TABLE campaigns (campaign_id VARCHAR, channel VARCHAR)")
    campaign_channels = {}
    campaign_rows = []
    for campaign_id, channel in iter_csv_rows(
        os.path.join(DATA, "campaigns.csv"), 2
    ):
        if campaign_id is not None:
            campaign_channels[campaign_id] = channel
            campaign_rows.append((campaign_id, channel))
    if campaign_rows:
        connection.executemany("INSERT INTO campaigns VALUES (?, ?)", campaign_rows)

    connection.execute(
        """
        CREATE TEMP TABLE event_raw (
            event_id VARCHAR, user_id VARCHAR, ts VARCHAR, ingest_ms BIGINT,
            event_type VARCHAR, product_id VARCHAR, price_cents VARCHAR
        )
        """
    )
    event_insert = "INSERT INTO event_raw VALUES (?, ?, ?, ?, ?, ?, ?)"

    def parsed_event_rows():
        for path in sorted(glob.glob(os.path.join(DATA, "events", "*.csv"))):
            for row in iter_csv_rows(path, EVENT_COLUMNS):
                event_id, user_id, timestamp, ingest_ts, event_type, product_id, price = row
                if event_id is None:
                    continue
                yield (
                    event_id,
                    user_id,
                    timestamp,
                    parse_ingest_timestamp(ingest_ts),
                    event_type,
                    product_id,
                    price,
                )

    insert_batches(connection, event_insert, parsed_event_rows())
    connection.execute(
        """
        CREATE TEMP TABLE event_dedup AS
        SELECT event_id, user_id, ts, ingest_ms, event_type, product_id, price_cents
        FROM event_raw
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY event_id ORDER BY ingest_ms
        ) = 1
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE events (
            person_id VARCHAR, ts_ms BIGINT, event_type VARCHAR,
            product_id VARCHAR, price_cents BIGINT
        )
        """
    )

    def validated_events():
        cursor = connection.execute(
            "SELECT user_id, ts, ingest_ms, event_type, product_id, price_cents "
            "FROM event_dedup"
        )
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            for user_id, timestamp, ingest_ms, event_type, product_id, price in rows:
                if user_id is None or product_id is None:
                    continue
                if event_type not in ("view", "cart", "purchase"):
                    continue
                event_ms = parse_timestamp_ms(timestamp)
                if (
                    event_ms is None
                    or event_ms < window_start_ms
                    or event_ms >= window_end_ms
                ):
                    continue
                if ingest_ms - event_ms > DAY_MS:
                    continue
                if event_type == "purchase":
                    if price is None or DIGITS_PATTERN.fullmatch(price) is None:
                        continue
                    price_value = int(price)
                    if price_value < 1 or price_value > 10_000_000:
                        continue
                else:
                    price_value = 0
                yield (
                    user_ids.person_name(user_id),
                    event_ms,
                    event_type,
                    product_id,
                    price_value,
                )

    insert_batches(
        connection,
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        validated_events(),
    )

    connection.execute(
        """
        CREATE TEMP TABLE click_raw (
            click_id VARCHAR, user_id VARCHAR, campaign_id VARCHAR, ts VARCHAR
        )
        """
    )

    def parsed_click_rows():
        for path in sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv"))):
            for click_id, user_id, campaign_id, timestamp in iter_csv_rows(
                path, CLICK_COLUMNS
            ):
                if click_id is not None:
                    yield (click_id, user_id, campaign_id, timestamp)

    insert_batches(
        connection,
        "INSERT INTO click_raw VALUES (?, ?, ?, ?)",
        parsed_click_rows(),
    )
    connection.execute(
        """
        CREATE TEMP TABLE click_dedup AS
        SELECT click_id, user_id, campaign_id, ts
        FROM click_raw
        QUALIFY ROW_NUMBER() OVER (PARTITION BY click_id) = 1
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE clicks (
            click_id VARCHAR, click_id_key BLOB, person_id VARCHAR,
            campaign_id VARCHAR, ts_ms BIGINT
        )
        """
    )

    def validated_clicks():
        cursor = connection.execute(
            "SELECT click_id, user_id, campaign_id, ts FROM click_dedup"
        )
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            for click_id, user_id, campaign_id, timestamp in rows:
                if (
                    user_id is None
                    or campaign_id not in campaign_channels
                ):
                    continue
                click_ms = parse_timestamp_ms(timestamp)
                if (
                    click_ms is None
                    or click_ms < window_start_ms - CLICK_WINDOW_MS
                    or click_ms >= window_end_ms
                ):
                    continue
                yield (
                    click_id,
                    click_id.encode("utf-8"),
                    user_ids.person_name(user_id),
                    campaign_id,
                    click_ms,
                )

    insert_batches(
        connection,
        "INSERT INTO clicks VALUES (?, ?, ?, ?, ?)",
        validated_clicks(),
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE session_events AS
        WITH ordered_events AS (
            SELECT *,
                   LAG(ts_ms) OVER (
                       PARTITION BY person_id ORDER BY ts_ms
                   ) AS previous_ts_ms
            FROM events
        ),
        session_boundaries AS (
            SELECT *,
                   CASE
                       WHEN previous_ts_ms IS NULL
                         OR ts_ms - previous_ts_ms > {SESSION_GAP_MS}
                       THEN 1 ELSE 0
                   END AS starts_session
            FROM ordered_events
        ),
        numbered_events AS (
            SELECT *,
                   SUM(starts_session) OVER (
                       PARTITION BY person_id ORDER BY ts_ms
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS session_number
            FROM session_boundaries
        )
        SELECT person_id, ts_ms, event_type, product_id, price_cents,
               session_number
        FROM numbered_events
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE sessions AS
        SELECT person_id, session_number,
               MIN(ts_ms) AS session_start_ms,
               CAST(epoch_ms(MIN(ts_ms)) AS DATE) AS day,
               COUNT(*)::BIGINT AS event_count,
               SUM(CASE WHEN event_type = 'purchase' THEN price_cents ELSE 0 END)::BIGINT
                   AS revenue_cents,
               SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)::BIGINT
                   AS purchase_count
        FROM session_events
        GROUP BY person_id, session_number
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE category_daily AS
        WITH category_totals AS (
            SELECT s.person_id, s.day,
                   COALESCE(p.category, 'unknown') AS category,
                   SUM(e.price_cents)::BIGINT AS category_revenue
            FROM session_events e
            JOIN sessions s
              ON s.person_id = e.person_id
             AND s.session_number = e.session_number
            LEFT JOIN products p ON p.product_id = e.product_id
            WHERE e.event_type = 'purchase'
            GROUP BY s.person_id, s.day, COALESCE(p.category, 'unknown')
        ),
        ranked_categories AS (
            SELECT person_id, day, category,
                   ROW_NUMBER() OVER (
                       PARTITION BY person_id, day
                       ORDER BY category_revenue DESC, category ASC
                   ) AS category_rank
            FROM category_totals
        )
        SELECT person_id, day, category
        FROM ranked_categories
        WHERE category_rank = 1
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE user_daily AS
        SELECT daily.person_id AS user_id, daily.day,
               daily.sessions, daily.events, daily.revenue_cents,
               categories.category AS top_category
        FROM (
            SELECT person_id, day,
                   COUNT(*)::BIGINT AS sessions,
                   SUM(event_count)::BIGINT AS events,
                   SUM(revenue_cents)::BIGINT AS revenue_cents
            FROM sessions
            GROUP BY person_id, day
        ) daily
        LEFT JOIN category_daily categories
          ON categories.person_id = daily.person_id
         AND categories.day = daily.day
        """
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE session_credit AS
        WITH eligible_clicks AS (
            SELECT s.person_id, s.session_number, c.campaign_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY s.person_id, s.session_number
                       ORDER BY c.ts_ms DESC, c.click_id_key ASC
                   ) AS click_rank
            FROM sessions s
            JOIN clicks c
              ON c.person_id = s.person_id
             AND c.ts_ms <= s.session_start_ms
             AND c.ts_ms >= s.session_start_ms - {CLICK_WINDOW_MS}
        )
        SELECT person_id, session_number, campaign_id
        FROM eligible_clicks
        WHERE click_rank = 1
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE session_assists AS
        SELECT DISTINCT s.person_id, s.session_number, c.campaign_id
        FROM sessions s
        JOIN clicks c
          ON c.person_id = s.person_id
         AND c.ts_ms <= s.session_start_ms
         AND c.ts_ms >= s.session_start_ms - {CLICK_WINDOW_MS}
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE session_campaign AS
        SELECT s.person_id, s.session_number, s.day, s.purchase_count,
               s.revenue_cents, COALESCE(c.campaign_id, 'organic') AS campaign_id
        FROM sessions s
        LEFT JOIN session_credit c
          ON c.person_id = s.person_id
         AND c.session_number = s.session_number
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE campaign_daily AS
        WITH credited AS (
            SELECT campaign_id, day,
                   COUNT(*)::BIGINT AS sessions,
                   SUM(CASE WHEN purchase_count > 0 THEN 1 ELSE 0 END)::BIGINT
                       AS purchasing_sessions,
                   SUM(revenue_cents)::BIGINT AS revenue_cents
            FROM session_campaign
            GROUP BY campaign_id, day
        ),
        assisted AS (
            SELECT a.campaign_id, s.day,
                   COUNT(*)::BIGINT AS assisted_sessions
            FROM session_assists a
            JOIN sessions s
              ON s.person_id = a.person_id
             AND s.session_number = a.session_number
            GROUP BY a.campaign_id, s.day
        ),
        combined AS (
            SELECT COALESCE(c.campaign_id, a.campaign_id) AS campaign_id,
                   COALESCE(c.day, a.day) AS day,
                   COALESCE(c.sessions, 0)::BIGINT AS sessions,
                   COALESCE(c.purchasing_sessions, 0)::BIGINT AS purchasing_sessions,
                   COALESCE(c.revenue_cents, 0)::BIGINT AS revenue_cents,
                   COALESCE(a.assisted_sessions, 0)::BIGINT AS assisted_sessions
            FROM credited c
            FULL OUTER JOIN assisted a
              ON a.campaign_id = c.campaign_id
             AND a.day = c.day
        )
        SELECT combined.campaign_id,
               COALESCE(campaigns.channel, 'organic') AS channel,
               combined.day, combined.sessions,
               combined.purchasing_sessions, combined.revenue_cents,
               combined.assisted_sessions
        FROM combined
        LEFT JOIN campaigns USING (campaign_id)
        """
    )

    user_output = os.path.join(OUT, "user_daily")
    campaign_output = os.path.join(OUT, "campaign_daily")
    os.makedirs(user_output, exist_ok=True)
    os.makedirs(campaign_output, exist_ok=True)

    connection.execute(
        f"""
        COPY (
            SELECT user_id::VARCHAR AS user_id, day::DATE AS day,
                   sessions::BIGINT AS sessions, events::BIGINT AS events,
                   revenue_cents::BIGINT AS revenue_cents,
                   top_category::VARCHAR AS top_category
            FROM user_daily
        ) TO {sql_string(os.path.join(user_output, 'part-0.parquet'))}
          (FORMAT PARQUET)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT campaign_id::VARCHAR AS campaign_id, channel::VARCHAR AS channel,
                   day::DATE AS day, sessions::BIGINT AS sessions,
                   purchasing_sessions::BIGINT AS purchasing_sessions,
                   revenue_cents::BIGINT AS revenue_cents,
                   assisted_sessions::BIGINT AS assisted_sessions
            FROM campaign_daily
        ) TO {sql_string(os.path.join(campaign_output, 'part-0.parquet'))}
          (FORMAT PARQUET)
        """
    )
    print("wrote user_daily and campaign_daily")


if __name__ == "__main__":
    main()
