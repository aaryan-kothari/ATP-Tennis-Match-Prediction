"""
tennis_fixtures.py — Fetches upcoming ATP match fixtures from The Odds API
(https://the-odds-api.com), with in-memory caching to stay well within the
free tier's 500 requests/month.

VERIFIED SCHEMA (confirmed against a real live API response on 2026-06-26):
    GET /v4/sports/?apiKey=...&all=true
        -> list of {key, group, title, description, active, has_outrights}
        Tennis sport keys are per-TOURNAMENT (e.g. "tennis_atp_wimbledon"),
        NOT a single generic "all ATP matches" key. Only tournaments
        currently in their event window have active=true; everything else
        sits at active=false until then.

    GET /v4/sports/{sport_key}/odds?apiKey=...&regions=uk&markets=h2h
        -> list of {id, sport_key, sport_title, commence_time,
                     home_team, away_team, bookmakers: [...]}
        home_team/away_team are FULL real names (e.g. "Naomi Osaka"), not the
        "Lastname F." format used in atp_tennis.csv. commence_time is
        ISO8601 UTC. There is NO round field and NO surface field in this
        response - both must come from elsewhere (see surface_lookup.py for
        surface; round is simply unavailable from this API and is left as
        None throughout this integration).

DESIGN NOTES:
- We do not infer surface from tournament name with fuzzy rules - we use the
  explicit surface_lookup.py table. Tournaments not in that table report
  surface=None, and callers must decide how to handle that (e.g. show
  "Surface: Unknown" in the UI, or let the user pick) rather than silently
  guessing Hard.
- Bookmaker odds in the API response are intentionally ignored - we only
  want the fixture (who's playing, when, where), not betting odds. Our own
  model is the one source of truth for win probabilities.
- Caching is in-memory with a TTL. This is fine for a single-process
  deployment (e.g. one Uvicorn worker). If this app is ever deployed with
  multiple worker processes, switch to a shared cache (Redis, etc.) so each
  worker doesn't independently burn quota - noted here so this doesn't bite
  someone later.
"""

import os
import time
import requests

from surface_lookup import lookup_surface

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

CACHE_TTL_SECONDS = 30 * 60  # 30 minutes, per the agreed design

# In-memory caches. _sports_cache holds the full /sports list (free to call,
# but no need to hammer it). _fixtures_cache holds odds-derived fixtures per
# sport_key (this one costs real quota, so the TTL matters a lot more here).
_sports_cache = {"data": None, "fetched_at": 0}
_fixtures_cache = {"data": None, "fetched_at": 0}


class OddsAPIError(Exception):
    """Raised when The Odds API can't be reached or returns an error we
    can't recover from. Callers (the FastAPI endpoint) should catch this and
    degrade gracefully rather than crash the whole request."""
    pass


def _check_api_key():
    if not ODDS_API_KEY:
        raise OddsAPIError(
            "ODDS_API_KEY environment variable is not set. "
            "Set it to your free API key from the-odds-api.com before "
            "calling any fixtures endpoint."
        )


def _get_active_tennis_sport_keys(force_refresh=False):
    """
    Returns the list of currently-active tennis sport_key dicts (men's ATP
    singles only - WTA keys are filtered out since this project predicts
    ATP outcomes). This call does NOT cost API quota per the docs.
    """
    _check_api_key()

    now = time.time()
    if not force_refresh and _sports_cache["data"] is not None:
        if now - _sports_cache["fetched_at"] < CACHE_TTL_SECONDS:
            return _sports_cache["data"]

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/",
            params={"apiKey": ODDS_API_KEY, "all": "true"},
            timeout=10,
        )
    except requests.RequestException as e:
        raise OddsAPIError(f"Could not reach The Odds API: {e}")

    if resp.status_code != 200:
        raise OddsAPIError(
            f"The Odds API returned {resp.status_code} on /sports: {resp.text[:300]}"
        )

    all_sports = resp.json()
    # ATP men's singles keys look like "tennis_atp_<tournament>". We exclude
    # WTA and any non-singles/doubles variants explicitly by key prefix
    # rather than by "description" text, since key prefixes are stable.
    atp_active = [
        s for s in all_sports
        if s.get("key", "").startswith("tennis_atp_") and s.get("active") is True
    ]

    _sports_cache["data"] = atp_active
    _sports_cache["fetched_at"] = now
    return atp_active


def _fetch_odds_for_sport_key(sport_key):
    """Single /odds call for one sport_key. Costs 1 request per region per
    market against quota - we request only 1 region (uk) and 1 market (h2h)
    to keep this as cheap as possible per call."""
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "uk", "markets": "h2h"},
            timeout=10,
        )
    except requests.RequestException as e:
        raise OddsAPIError(f"Could not reach The Odds API for {sport_key}: {e}")

    if resp.status_code == 401:
        raise OddsAPIError("The Odds API rejected the API key (401 Unauthorized). Check ODDS_API_KEY.")
    if resp.status_code == 429:
        raise OddsAPIError("The Odds API quota exhausted (429 Too Many Requests) - try again next month, or upgrade the plan.")
    if resp.status_code != 200:
        raise OddsAPIError(f"The Odds API returned {resp.status_code} for {sport_key}: {resp.text[:300]}")

    return resp.json()


def get_upcoming_fixtures(force_refresh=False):
    """
    Returns a normalized list of upcoming ATP fixtures:
        [
          {
            "fixture_id": str,
            "tournament": str,           # e.g. "ATP Wimbledon"
            "player_a": str,              # full name, e.g. "Jannik Sinner"
            "player_b": str,
            "commence_time": str,         # ISO8601 UTC, as given by the API
            "surface": str or None,       # from surface_lookup table; None if unknown
            "round": None,                # always None - not available from this API
          },
          ...
        ]

    Results are cached for CACHE_TTL_SECONDS. Pass force_refresh=True to
    bypass the cache (use sparingly - this costs real API quota).

    Raises OddsAPIError if the API key is missing or the API can't be
    reached - callers should catch this and respond with a clear error
    rather than letting it propagate as a generic 500.
    """
    now = time.time()
    if not force_refresh and _fixtures_cache["data"] is not None:
        if now - _fixtures_cache["fetched_at"] < CACHE_TTL_SECONDS:
            return _fixtures_cache["data"]

    active_sports = _get_active_tennis_sport_keys(force_refresh=force_refresh)

    fixtures = []
    errors = []
    for sport in active_sports:
        sport_key = sport["key"]
        tournament_title = sport.get("title", sport_key)
        try:
            matches = _fetch_odds_for_sport_key(sport_key)
        except OddsAPIError as e:
            # One tournament failing shouldn't take down the whole fixtures
            # list - record the error and keep going with the others.
            errors.append(str(e))
            continue

        for m in matches:
            fixtures.append({
                "fixture_id": m.get("id"),
                "tournament": tournament_title,
                "player_a": m.get("home_team"),
                "player_b": m.get("away_team"),
                "commence_time": m.get("commence_time"),
                "surface": lookup_surface(tournament_title),
                "round": None,  # not available from this API - see module docstring
            })

    # Sort soonest-first so the frontend doesn't have to
    fixtures.sort(key=lambda f: f["commence_time"] or "")

    _fixtures_cache["data"] = fixtures
    _fixtures_cache["fetched_at"] = now

    if not fixtures and errors:
        # Every tournament call failed and we have nothing to show - this
        # should surface as a real error, not a silent empty list.
        raise OddsAPIError("; ".join(errors))

    return fixtures


def cache_status():
    """For the /health-style endpoint - shows cache freshness without
    triggering a refresh."""
    now = time.time()
    return {
        "sports_cache_age_seconds": (
            round(now - _sports_cache["fetched_at"]) if _sports_cache["data"] is not None else None
        ),
        "fixtures_cache_age_seconds": (
            round(now - _fixtures_cache["fetched_at"]) if _fixtures_cache["data"] is not None else None
        ),
        "fixtures_cached_count": len(_fixtures_cache["data"]) if _fixtures_cache["data"] is not None else 0,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


if __name__ == "__main__":
    import json
    try:
        result = get_upcoming_fixtures()
        print(f"Found {len(result)} upcoming ATP fixtures:\n")
        print(json.dumps(result, indent=2))
    except OddsAPIError as e:
        print(f"OddsAPIError: {e}")
