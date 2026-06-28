"""
outcome_resolver.py — Checks The Odds API's /scores endpoint for completed
matches and updates tracking_db with the real outcome, so predictions can
be graded.

IMPORTANT CAVEAT (documented here because it affects correctness, not just
performance): The Odds API's own docs state "The scores endpoint applies to
selected sports and is gradually being expanded to more sports." Tennis
coverage was NOT directly confirmed against a live call at the time this
was written - this module is built defensively so that if tennis scores
are unavailable or come back in an unexpected shape, predictions are marked
'unresolved' rather than silently marked wrong or crashing the poll loop.
If you confirm tennis /scores works differently than assumed here, the
parsing logic in _extract_winner() is the one place to adjust.

SCHEMA ASSUMED (from the documented /scores response, generalized - not
tennis-specific in the docs, but tennis should follow the same shape):
    {
      "id": "...",
      "sport_key": "tennis_atp_wimbledon",
      "commence_time": "...",
      "completed": true,
      "home_team": "...",
      "away_team": "...",
      "scores": [
        {"name": "...", "score": "<higher score = winner>"},
        {"name": "...", "score": "..."}
      ],
      "last_update": "..."
    }
"completed": false or scores: null means no usable result yet.

RESOLUTION WINDOW: The Odds API's daysFrom parameter only returns games
completed in the last 1-3 days. We use daysFrom=3 (the max) to give the
widest window. A prediction whose match happened more than 3 days ago and
was never found gets marked 'unresolved' by check_and_resolve_stale() so it
doesn't sit as 'pending' forever.
"""

import os
import time
from datetime import datetime, timezone, timedelta

import requests

import tracking_db as db

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

STALE_AFTER_DAYS = 4  # if a match commenced this long ago and we still have
                       # no result, stop waiting and mark unresolved


class OutcomeResolverError(Exception):
    pass


def _fetch_scores_for_sport_key(sport_key, days_from=3):
    if not ODDS_API_KEY:
        raise OutcomeResolverError("ODDS_API_KEY is not set.")
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/scores/",
            params={"apiKey": ODDS_API_KEY, "daysFrom": days_from},
            timeout=10,
        )
    except requests.RequestException as e:
        raise OutcomeResolverError(f"Could not reach The Odds API scores for {sport_key}: {e}")

    if resp.status_code != 200:
        raise OutcomeResolverError(
            f"The Odds API returned {resp.status_code} for {sport_key} scores: {resp.text[:300]}"
        )
    return resp.json()


def _extract_winner(score_event):
    """
    Given one event from /scores, returns the winning team's name (matching
    home_team/away_team strings) if determinable, else None.

    Defensive against: completed=False, scores=null, malformed/missing
    score values, and a tie (which shouldn't happen in tennis - no draws -
    but we don't assume that and just return None rather than guess).
    """
    if not score_event.get("completed"):
        return None
    scores = score_event.get("scores")
    if not scores or len(scores) != 2:
        return None

    try:
        s0 = float(scores[0]["score"])
        s1 = float(scores[1]["score"])
    except (KeyError, TypeError, ValueError):
        return None

    if s0 == s1:
        return None  # no tiebreak info available - can't determine a winner safely

    return scores[0]["name"] if s0 > s1 else scores[1]["name"]


def _match_api_name_to_resolved(api_name, pred_row, player_resolver):
    """
    The /scores response gives names in The Odds API's own format (full
    names), same as /odds. We need to map that back to OUR dataset's
    resolved format ("Sinner J.") to compare against predicted_winner,
    which is stored in resolved format. We reuse predict.py's own resolver
    rather than re-implementing name matching here - single source of
    truth for "what does this name refer to".

    As a safety net: if the resolver can't place it, we fall back to
    checking it against the resolved_player_a/b already stored on this
    exact prediction row (cheap, since those are already known correct for
    this fixture from when the prediction was made).
    """
    try:
        resolved = player_resolver(api_name)
        if resolved:
            return resolved
    except ValueError:
        pass  # ambiguous via the general resolver - fall back below

    # Fallback: does the api_name loosely match one of the two players
    # already on this row? (handles minor formatting drift between the
    # /odds and /scores responses for the same player)
    lower = api_name.lower()
    if pred_row["player_a"].lower() == lower or pred_row["resolved_player_a"].lower() in lower:
        return pred_row["resolved_player_a"]
    if pred_row["player_b"].lower() == lower or pred_row["resolved_player_b"].lower() in lower:
        return pred_row["resolved_player_b"]
    return None


def resolve_pending_predictions(player_resolver, active_tennis_sport_keys):
    """
    Main entry point. For every pending tracked prediction, tries to find
    its real outcome via The Odds API scores, and updates tracking_db
    accordingly.

    Parameters
    ----------
    player_resolver : callable(str) -> str | None
        A function that converts a full player name into the dataset's
        resolved format. In production this is
        predict.stats_store.resolve_player_name.
    active_tennis_sport_keys : list[str]
        Sport keys to check (e.g. ["tennis_atp_wimbledon"]) - normally the
        currently-active ATP keys from tennis_fixtures.py.

    Returns a summary dict for logging/observability.
    """
    pending = db.get_pending_predictions()
    if not pending:
        return {"checked": 0, "resolved": 0, "unresolved": 0, "still_pending": 0, "errors": []}

    pending_by_id = {p["fixture_id"]: p for p in pending}
    resolved_count = 0
    unresolved_count = 0
    errors = []

    for sport_key in active_tennis_sport_keys:
        try:
            scores_events = _fetch_scores_for_sport_key(sport_key)
        except OutcomeResolverError as e:
            errors.append(str(e))
            continue

        for event in scores_events:
            fixture_id = event.get("id")
            if fixture_id not in pending_by_id:
                continue  # not one of ours, or already resolved

            pred_row = pending_by_id[fixture_id]
            winner_api_name = _extract_winner(event)

            if winner_api_name is None:
                continue  # not completed yet, or no usable score - leave pending

            resolved_winner = _match_api_name_to_resolved(winner_api_name, pred_row, player_resolver)
            if resolved_winner is None:
                # Match completed but we couldn't figure out who won in our
                # naming scheme - mark unresolved rather than silently drop it.
                db.mark_unresolved(fixture_id)
                unresolved_count += 1
                del pending_by_id[fixture_id]
                continue

            predicted_winner = pred_row["predicted_winner"]
            if predicted_winner is None:
                was_correct = None  # "Too Close to Call" - nothing to grade
            else:
                was_correct = 1 if predicted_winner == resolved_winner else 0

            db.mark_resolved(fixture_id, actual_winner=resolved_winner, was_correct=was_correct)
            resolved_count += 1
            del pending_by_id[fixture_id]

    return {
        "checked": len(pending),
        "resolved": resolved_count,
        "unresolved": unresolved_count,
        "still_pending": len(pending_by_id),
        "errors": errors,
    }


def mark_stale_predictions_unresolved():
    """
    Sweeps remaining pending predictions whose commence_time is older than
    STALE_AFTER_DAYS and marks them unresolved - these fell outside the
    scores API's lookback window and will never resolve automatically.
    Call this periodically (e.g. once a day) alongside the main resolver.
    """
    pending = db.get_pending_predictions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)
    marked = 0

    for p in pending:
        commence_time = p.get("commence_time")
        if not commence_time:
            continue
        try:
            commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        if commence_dt < cutoff:
            db.mark_unresolved(p["fixture_id"])
            marked += 1

    return {"marked_stale": marked}
