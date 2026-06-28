"""
tracking_db.py — SQLite-backed storage for prediction tracking and accuracy
reporting.

Only predictions made against a REAL fixture_id (from /fixtures/upcoming)
are tracked here. Ad-hoc predictions (manually typed player names with no
fixture_id) are never logged - there's no future real-world outcome to
verify them against, so "accuracy" would be meaningless for those.

Schema (single table, kept deliberately simple):

    tracked_predictions
        fixture_id        TEXT PRIMARY KEY   -- from The Odds API, stable per match
        tournament         TEXT
        surface             TEXT
        player_a            TEXT              -- as given to /predict (API format)
        player_b            TEXT
        resolved_player_a   TEXT              -- dataset format, e.g. "Sinner J."
        resolved_player_b   TEXT
        prob_a_wins         REAL
        prob_b_wins         REAL
        predicted_winner    TEXT               -- resolved_player_a/b, or NULL if "Too Close to Call"
        confidence          TEXT
        predicted_at        TEXT               -- ISO8601 UTC timestamp
        commence_time       TEXT               -- from the fixture, ISO8601 UTC
        status              TEXT               -- 'pending' | 'completed' | 'unresolved'
        actual_winner       TEXT               -- resolved_player_a/b once known, else NULL
        was_correct         INTEGER            -- 1/0/NULL (NULL while pending or "too close")
        resolved_at         TEXT               -- ISO8601 UTC timestamp of when we found the result

A prediction with confidence "Too Close to Call" (predicted_winner IS NULL)
is still stored and still gets its actual_winner filled in once known, but
is excluded from "correct/incorrect" counts in the accuracy report - there
was no call made, so there's nothing to grade as right or wrong. It's
included in a separate "too_close_count" instead.

'unresolved' status means we tried to look up the result and either the
match was found but had no usable score data, or we never found it within
a reasonable window (e.g. The Odds API's scores endpoint only covers a
rolling few days - see tennis_fixtures.py / outcome_resolver.py for the
specific window used). Unresolved predictions are excluded from accuracy
stats entirely (not counted as wrong) and shown as a distinct count so the
dashboard is honest about what it couldn't verify.
"""

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "TENNIS_TRACKING_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracked_predictions (
    fixture_id        TEXT PRIMARY KEY,
    tournament        TEXT NOT NULL,
    surface           TEXT NOT NULL,
    player_a          TEXT NOT NULL,
    player_b          TEXT NOT NULL,
    resolved_player_a TEXT NOT NULL,
    resolved_player_b TEXT NOT NULL,
    prob_a_wins       REAL NOT NULL,
    prob_b_wins       REAL NOT NULL,
    predicted_winner  TEXT,
    confidence        TEXT NOT NULL,
    predicted_at      TEXT NOT NULL,
    commence_time     TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    actual_winner     TEXT,
    was_correct       INTEGER,
    resolved_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_status ON tracked_predictions(status);
CREATE INDEX IF NOT EXISTS idx_tournament ON tracked_predictions(tournament);
CREATE INDEX IF NOT EXISTS idx_surface ON tracked_predictions(surface);
"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection():
    """Context-managed connection - commits on clean exit, rolls back on
    exception, always closes. Use this everywhere rather than opening raw
    connections, so we never leak a connection or leave a half-written
    transaction on a crash mid-write."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates the table/indexes if they don't already exist. Safe to call
    on every app startup - CREATE TABLE IF NOT EXISTS is a no-op if it's
    already there."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


def record_prediction(
    fixture_id, tournament, surface, player_a, player_b,
    resolved_player_a, resolved_player_b, prob_a_wins, prob_b_wins,
    predicted_winner, confidence, commence_time=None,
):
    """
    Inserts a new tracked prediction, or REPLACES the existing row if this
    fixture_id was already predicted before (e.g. someone clicked Predict
    twice on the same match) - we only ever want the most recent prediction
    per fixture, not duplicate/stale ones. Resets status back to 'pending'
    on overwrite since this is logically a fresh prediction.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tracked_predictions (
                fixture_id, tournament, surface, player_a, player_b,
                resolved_player_a, resolved_player_b, prob_a_wins, prob_b_wins,
                predicted_winner, confidence, predicted_at, commence_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(fixture_id) DO UPDATE SET
                tournament=excluded.tournament,
                surface=excluded.surface,
                player_a=excluded.player_a,
                player_b=excluded.player_b,
                resolved_player_a=excluded.resolved_player_a,
                resolved_player_b=excluded.resolved_player_b,
                prob_a_wins=excluded.prob_a_wins,
                prob_b_wins=excluded.prob_b_wins,
                predicted_winner=excluded.predicted_winner,
                confidence=excluded.confidence,
                predicted_at=excluded.predicted_at,
                commence_time=excluded.commence_time,
                status='pending',
                actual_winner=NULL,
                was_correct=NULL,
                resolved_at=NULL
            """,
            (
                fixture_id, tournament, surface, player_a, player_b,
                resolved_player_a, resolved_player_b, prob_a_wins, prob_b_wins,
                predicted_winner, confidence, _now_iso(), commence_time,
            ),
        )


def get_pending_predictions():
    """Returns all predictions still awaiting an outcome - this is the set
    the outcome resolver needs to check against The Odds API's /scores."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tracked_predictions WHERE status = 'pending'"
        ).fetchall()
        return [dict(r) for r in rows]


def mark_resolved(fixture_id, actual_winner, was_correct):
    """Marks a prediction as completed with a known outcome. was_correct is
    None when the prediction itself had no call to grade (confidence was
    'Too Close to Call', predicted_winner was NULL)."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tracked_predictions
            SET status = 'completed', actual_winner = ?, was_correct = ?, resolved_at = ?
            WHERE fixture_id = ?
            """,
            (actual_winner, was_correct, _now_iso(), fixture_id),
        )


def mark_unresolved(fixture_id):
    """Marks a prediction as unresolved - we looked but couldn't determine
    a real outcome (e.g. match was walkover/retired with no clean winner,
    or it fell outside the scores API's lookback window)."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tracked_predictions
            SET status = 'unresolved', resolved_at = ?
            WHERE fixture_id = ?
            """,
            (_now_iso(), fixture_id),
        )


def get_accuracy_report():
    """
    Computes the full accuracy report from completed predictions:
    overall accuracy, per-tournament accuracy, per-surface accuracy, and
    correct/incorrect/too-close/pending/unresolved counts.

    "Accuracy" is always computed only over predictions where a real call
    was made (predicted_winner IS NOT NULL) AND the outcome is known
    (status = 'completed') - "Too Close to Call" predictions and
    pending/unresolved ones are reported separately, never silently folded
    into the accuracy denominator.
    """
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM tracked_predictions").fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_predictions WHERE status = 'pending'"
        ).fetchone()["n"]
        unresolved = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_predictions WHERE status = 'unresolved'"
        ).fetchone()["n"]
        too_close = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_predictions "
            "WHERE status = 'completed' AND predicted_winner IS NULL"
        ).fetchone()["n"]

        graded = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_predictions "
            "WHERE status = 'completed' AND predicted_winner IS NOT NULL"
        ).fetchone()["n"]
        correct = conn.execute(
            "SELECT COUNT(*) AS n FROM tracked_predictions "
            "WHERE status = 'completed' AND predicted_winner IS NOT NULL AND was_correct = 1"
        ).fetchone()["n"]

        overall_accuracy = (correct / graded) if graded > 0 else None

        by_tournament = conn.execute(
            """
            SELECT tournament,
                   COUNT(*) AS graded,
                   SUM(was_correct) AS correct
            FROM tracked_predictions
            WHERE status = 'completed' AND predicted_winner IS NOT NULL
            GROUP BY tournament
            ORDER BY graded DESC
            """
        ).fetchall()

        by_surface = conn.execute(
            """
            SELECT surface,
                   COUNT(*) AS graded,
                   SUM(was_correct) AS correct
            FROM tracked_predictions
            WHERE status = 'completed' AND predicted_winner IS NOT NULL
            GROUP BY surface
            ORDER BY graded DESC
            """
        ).fetchall()

        by_confidence = conn.execute(
            """
            SELECT confidence,
                   COUNT(*) AS graded,
                   SUM(was_correct) AS correct
            FROM tracked_predictions
            WHERE status = 'completed' AND predicted_winner IS NOT NULL
            GROUP BY confidence
            ORDER BY graded DESC
            """
        ).fetchall()

        recent = conn.execute(
            """
            SELECT * FROM tracked_predictions
            WHERE status = 'completed'
            ORDER BY resolved_at DESC
            LIMIT 20
            """
        ).fetchall()

    def _rate(row):
        g = row["graded"]
        c = row["correct"] or 0
        return {
            "graded": g,
            "correct": c,
            "incorrect": g - c,
            "accuracy": round(c / g, 4) if g > 0 else None,
        }

    return {
        "total_tracked": total,
        "pending": pending,
        "unresolved": unresolved,
        "too_close_to_call": too_close,
        "graded": graded,
        "correct": correct,
        "incorrect": graded - correct,
        "overall_accuracy": round(overall_accuracy, 4) if overall_accuracy is not None else None,
        "by_tournament": {row["tournament"]: _rate(row) for row in by_tournament},
        "by_surface": {row["surface"]: _rate(row) for row in by_surface},
        "by_confidence": {row["confidence"]: _rate(row) for row in by_confidence},
        "recent_results": [dict(r) for r in recent],
    }


def get_all_predictions(limit=200):
    """Returns all tracked predictions, most recent first - used for a
    detailed/debug view rather than the summary report."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tracked_predictions ORDER BY predicted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
