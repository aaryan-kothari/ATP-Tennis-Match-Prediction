"""
app.py — FastAPI backend for the tennis match prediction engine.

Wraps predict.py's predict_match() function as a REST API, plus a live
upcoming-fixtures feed from The Odds API (see tennis_fixtures.py) and a
prediction accuracy dashboard backed by SQLite (see tracking_db.py /
outcome_resolver.py).

Run locally with:
    uvicorn app:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API documentation
(FastAPI generates this automatically - no extra work needed).

Before running, set the ODDS_API_KEY environment variable to your free key
from the-odds-api.com (needed for /fixtures/upcoming AND for the dashboard's
automatic outcome resolution - /predict and the player-lookup endpoints work
without it).

Endpoints:
    GET  /health                - basic liveness check + dataset stats
    GET  /players/search        - autocomplete/typeahead for player names
    GET  /surfaces                - list valid surface values
    GET  /tournament-options      - list valid series/round values
    POST /predict                 - the main prediction endpoint
    GET  /fixtures/upcoming       - live upcoming ATP matches (cached)
    GET  /dashboard/accuracy      - prediction accuracy report
    GET  /dashboard/predictions   - raw tracked-prediction list (debug view)
    POST /dashboard/sync          - manually trigger outcome resolution
"""

from dotenv import load_dotenv

load_dotenv()
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from predict import (
    predict_match,
    stats_store,
    _KNOWN_SERIES,
    _KNOWN_ROUNDS,
)
from tennis_fixtures import get_upcoming_fixtures, cache_status, OddsAPIError, _get_active_tennis_sport_keys
import tracking_db as db
import outcome_resolver

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Tennis Match Outcome Prediction API",
    description="Predicts ATP match win probabilities using historical "
                "stats, Elo ratings, head-to-head records, and recent form.",
    version="1.0.0",
)

# CORS: allow a frontend running on a different origin (e.g. localhost:3000
# for a React dev server) to call this API. Tighten allow_origins to your
# actual frontend domain before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    player_a: str = Field(..., examples=["Sinner J."], description="First player's name")
    player_b: str = Field(..., examples=["Alcaraz C."], description="Second player's name")
    surface: str = Field(..., examples=["Hard"], description="Hard, Clay, or Grass")
    series: Optional[str] = Field(None, examples=["Grand Slam"], description="Tournament category (optional)")
    round: Optional[str] = Field(None, examples=["Quarterfinals"], description="Match round (optional)")
    best_of: Literal[3, 5] = Field(3, description="3 or 5 sets (no other value is valid in tennis)")
    fixture_id: Optional[str] = Field(
        None,
        description="If this prediction is for a real fixture from /fixtures/upcoming, "
                     "pass its fixture_id here to have it tracked on the accuracy "
                     "dashboard. Omit for ad-hoc/manual predictions - those are never "
                     "tracked, since there's no real future outcome to verify them "
                     "against.",
    )
    tournament: Optional[str] = Field(
        None, description="Tournament name - required alongside fixture_id for tracking "
                           "(used for the dashboard's per-tournament breakdown)."
    )


class PredictResponse(BaseModel):
    player_a: str
    player_b: str
    resolved_player_a: str
    resolved_player_b: str
    surface: str
    series: Optional[str]
    round: Optional[str]
    best_of: int
    prob_a_wins: float
    prob_b_wins: float
    predicted_winner: Optional[str]
    confidence: str
    notes: List[str]
    tracked: bool = Field(
        False, description="True if this prediction was logged to the "
                            "accuracy dashboard (only happens when fixture_id was provided)."
    )


class FixtureResponse(BaseModel):
    fixture_id: Optional[str]
    tournament: str
    player_a: str
    player_b: str
    commence_time: Optional[str]
    surface: str = Field(
        description="The surface to use for prediction. If surface_known is "
                     "false, this defaults to 'Hard' and the frontend should "
        "show a surface selector rather than treating this as confirmed."
    )
    surface_known: bool = Field(
        description="True if the surface came from the known-tournament "
                     "lookup table; false if it's a default guess and the "
                     "user should be given the option to override it."
    )
    round: Optional[str] = Field(
        None, description="Always null - round/stage data is not available "
                           "from the current fixtures source."
    )


class AccuracyBucket(BaseModel):
    graded: int
    correct: int
    incorrect: int
    accuracy: Optional[float]


class AccuracyReport(BaseModel):
    total_tracked: int
    pending: int
    unresolved: int
    too_close_to_call: int
    graded: int
    correct: int
    incorrect: int
    overall_accuracy: Optional[float]
    by_tournament: dict
    by_surface: dict
    by_confidence: dict
    recent_results: List[dict]


class SyncResult(BaseModel):
    checked: int
    resolved: int
    unresolved: int
    still_pending: int
    errors: List[str]
    stale_marked: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Basic liveness check - also confirms the dataset loaded correctly,
    and shows fixtures-cache freshness (does not trigger a refresh)."""
    return {
        "status": "ok",
        "matches_loaded": len(stats_store.df),
        "distinct_players": len(stats_store.known_players),
        "latest_match_date": str(stats_store.last_seen_date.date()),
        "fixtures_cache": cache_status(),
    }


@app.get("/players/search")
def search_players(
    q: str = Query(..., min_length=1, description="Partial player name to search for"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Autocomplete/typeahead endpoint for a frontend search box. Returns up to
    `limit` matching player names - does not raise on ambiguity, always
    returns a (possibly empty) list.
    """
    matches = stats_store.search_players(q, limit=limit)
    return {"query": q, "matches": matches}


@app.get("/surfaces")
def list_surfaces():
    """Valid surface values, derived from the training data."""
    return {"surfaces": sorted(stats_store.df["Surface"].dropna().unique().tolist())}


@app.get("/tournament-options")
def tournament_options():
    """Valid Series and Round values the model recognizes (matches the
    one-hot columns baked into the trained model)."""
    return {"series": _KNOWN_SERIES, "rounds": _KNOWN_ROUNDS}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict the outcome of a match between two players.

    Returns win probabilities for both players plus a confidence label:
    High (>=70%), Medium (60-70%), Low (55-60%), or "Too Close to Call"
    (45-55%, in which case `predicted_winner` is null rather than guessing).

    If `fixture_id` is provided (i.e. this prediction is for a real match
    from /fixtures/upcoming, not an ad-hoc manual query), the prediction is
    logged to the accuracy dashboard for later grading once the real result
    is known.
    """
    try:
        result = predict_match(
            player_a=request.player_a,
            player_b=request.player_b,
            surface=request.surface,
            series=request.series,
            round_=request.round,
            best_of=request.best_of,
        )
    except ValueError as e:
        # Ambiguous player name (matched multiple players) - this is a client
        # error (400), not a server error, since the request itself needs
        # clarification from the caller.
        raise HTTPException(status_code=400, detail=str(e))

    tracked = False
    if request.fixture_id:
        try:
            # IMPORTANT: predict_match()'s `predicted_winner` is in the ORIGINAL
            # input format (e.g. "Jannik Sinner"), kept that way deliberately
            # for human-readable API responses. The tracking DB instead needs
            # the RESOLVED dataset format (e.g. "Sinner J.") here, because
            # outcome_resolver.py always compares against resolved names when
            # grading - storing the unresolved form would make every single
            # comparison silently fail (verified during testing: this was
            # actually wrong on the first pass, caught by checking a known-
            # correct prediction came back marked incorrect).
            if result["predicted_winner"] is None:
                predicted_winner_resolved = None
            elif result["predicted_winner"] == request.player_a:
                predicted_winner_resolved = result["resolved_player_a"]
            else:
                predicted_winner_resolved = result["resolved_player_b"]

            db.record_prediction(
                fixture_id=request.fixture_id,
                tournament=request.tournament or "Unknown",
                surface=request.surface,
                player_a=request.player_a,
                player_b=request.player_b,
                resolved_player_a=result["resolved_player_a"],
                resolved_player_b=result["resolved_player_b"],
                prob_a_wins=result["prob_a_wins"],
                prob_b_wins=result["prob_b_wins"],
                predicted_winner=predicted_winner_resolved,
                confidence=result["confidence"],
            )
            tracked = True
        except Exception as e:
            # Tracking is a side-effect, not the primary purpose of this
            # endpoint - a DB hiccup should never prevent the caller from
            # getting their prediction. Log it, don't fail the request.
            logger.warning(f"Failed to record tracked prediction for {request.fixture_id}: {e}")

    result["tracked"] = tracked
    return result


@app.get("/fixtures/upcoming", response_model=List[FixtureResponse])
def fixtures_upcoming(force_refresh: bool = Query(False, description="Bypass cache - costs real API quota, use sparingly")):
    """
    Live upcoming ATP fixtures from The Odds API, normalized for display.

    Results are cached for 30 minutes server-side to conserve the free-tier
    quota (500 requests/month) - repeated calls within that window return
    the same cached data and cost nothing.

    surface_known=false means the tournament wasn't in our lookup table; the
    'surface' field defaults to "Hard" in that case but should be treated as
    a guess, not a fact - the frontend should offer a surface selector for
    those fixtures rather than silently predicting on an assumed surface.

    Round/stage information (e.g. "Round 1", "Quarterfinal") is NOT
    available from this data source and is always null.
    """
    try:
        fixtures = get_upcoming_fixtures(force_refresh=force_refresh)
    except OddsAPIError as e:
        # Distinguish a missing/bad key (misconfiguration, our fault) from a
        # genuine upstream outage/quota issue (their fault) so whoever's
        # debugging this knows where to look.
        message = str(e)
        if "ODDS_API_KEY" in message or "401" in message:
            raise HTTPException(status_code=500, detail=f"Fixtures API misconfigured: {message}")
        raise HTTPException(status_code=503, detail=f"Fixtures API unavailable: {message}")

    response = []
    for f in fixtures:
        known = f["surface"] is not None
        response.append({
            "fixture_id": f["fixture_id"],
            "tournament": f["tournament"],
            "player_a": f["player_a"],
            "player_b": f["player_b"],
            "commence_time": f["commence_time"],
            "surface": f["surface"] if known else "Hard",
            "surface_known": known,
            "round": f["round"],
        })
    return response


@app.get("/dashboard/accuracy", response_model=AccuracyReport)
def dashboard_accuracy():
    """
    Prediction accuracy report: overall accuracy, plus breakdowns by
    tournament, surface, and confidence level. Only predictions tied to a
    real fixture_id are included (see /predict's fixture_id parameter) -
    ad-hoc manual predictions are never tracked, so they never appear here.

    'graded' predictions are ones where a real call was made (confidence
    was not "Too Close to Call") AND the real outcome is now known.
    'too_close_to_call' and 'pending'/'unresolved' predictions are reported
    separately and excluded from the accuracy percentage, since there's
    nothing to grade them against (either no call was made, or no outcome
    is known yet).
    """
    return db.get_accuracy_report()


@app.get("/dashboard/predictions")
def dashboard_predictions(limit: int = Query(200, ge=1, le=1000)):
    """Raw list of tracked predictions, most recent first - a detail/debug
    view behind the summary accuracy report."""
    return {"predictions": db.get_all_predictions(limit=limit)}


@app.post("/dashboard/sync", response_model=SyncResult)
def dashboard_sync():
    """
    Manually triggers outcome resolution immediately, rather than waiting
    for the next automatic background poll. Useful right after a match you
    know just finished, or for testing. Safe to call repeatedly - it's a
    no-op (zero quota cost beyond the /scores calls themselves) if there's
    nothing pending.
    """
    try:
        active_keys = [s["key"] for s in _get_active_tennis_sport_keys()]
    except OddsAPIError as e:
        raise HTTPException(status_code=503, detail=f"Could not determine active tournaments: {e}")

    result = outcome_resolver.resolve_pending_predictions(
        player_resolver=stats_store.resolve_player_name,
        active_tennis_sport_keys=active_keys,
    )
    stale_result = outcome_resolver.mark_stale_predictions_unresolved()
    result["stale_marked"] = stale_result["marked_stale"]
    return result


# ---------------------------------------------------------------------------
# Background scheduler - automatic outcome resolution
# ---------------------------------------------------------------------------
_scheduler = BackgroundScheduler()


def _scheduled_sync_job():
    """Runs on a timer in the background - same logic as POST /dashboard/sync
    but invoked automatically rather than by a client request. Wrapped in a
    broad try/except because an uncaught exception inside a scheduled job
    would otherwise silently kill that job's future runs without any clear
    error in the logs."""
    try:
        active_keys = [s["key"] for s in _get_active_tennis_sport_keys()]
        result = outcome_resolver.resolve_pending_predictions(
            player_resolver=stats_store.resolve_player_name,
            active_tennis_sport_keys=active_keys,
        )
        stale_result = outcome_resolver.mark_stale_predictions_unresolved()
        logger.info(
            f"[scheduled sync] resolved={result['resolved']} "
            f"unresolved={result['unresolved']} still_pending={result['still_pending']} "
            f"stale_marked={stale_result['marked_stale']} errors={result['errors']}"
        )
    except Exception as e:
        logger.warning(f"[scheduled sync] failed: {e}")


@app.on_event("startup")
def _on_startup():
    db.init_db()
    # Poll every 30 minutes - frequent enough to keep the dashboard
    # reasonably fresh, infrequent enough to stay well within the free
    # tier's monthly quota alongside the fixtures-polling cost.
    _scheduler.add_job(_scheduled_sync_job, "interval", minutes=30, id="outcome_sync")
    _scheduler.start()
    logger.info("Background outcome-sync scheduler started (every 30 minutes).")


@app.on_event("shutdown")
def _on_shutdown():
    _scheduler.shutdown(wait=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)