"""
surface_lookup.py — Known tournament -> surface mapping.

Per design decision: we do NOT infer surface from tournament name patterns or
guess defensively. We maintain an explicit lookup table for tournaments we
know, and anything not in the table is returned as None / "Unknown" so the
caller can decide how to handle it (e.g. ask the frontend user to pick a
surface, or simply omit surface-dependent features rather than guessing
"Hard" and silently degrading prediction quality for ATP 250/500 events that
are just as likely to be Clay or Indoor Hard).

This table is deliberately small and explicit. Extend it as needed - do not
add fuzzy/pattern-based matching here without a real reason, since silent
wrong guesses are worse than an honest "Unknown".
"""

# Grand Slams
GRAND_SLAM_SURFACES = {
    "wimbledon": "Grass",
    "french open": "Clay",
    "roland garros": "Clay",
    "australian open": "Hard",
    "us open": "Hard",
}

# ATP Masters 1000 (surfaces are stable year to year for these)
MASTERS_1000_SURFACES = {
    "indian wells": "Hard",
    "miami open": "Hard",
    "monte carlo masters": "Clay",
    "monte-carlo masters": "Clay",
    "madrid open": "Clay",
    "italian open": "Clay",
    "rome masters": "Clay",
    "canadian open": "Hard",
    "cincinnati masters": "Hard",
    "shanghai masters": "Hard",
    "paris masters": "Hard",  # indoor hard
}

# Merge into one lookup; keys are lowercased for matching
_ALL_KNOWN = {**GRAND_SLAM_SURFACES, **MASTERS_1000_SURFACES}


def lookup_surface(tournament_name: str):
    """
    Returns the known surface for a tournament name, or None if not in our
    table. Matching is case-insensitive and tolerant of the tournament name
    appearing as a substring (e.g. "Wimbledon 2026" or "ATP Wimbledon").

    Returns
    -------
    str or None - the surface ("Hard", "Clay", "Grass") if known, else None.
    """
    if not tournament_name:
        return None
    lower = tournament_name.lower()
    for key, surface in _ALL_KNOWN.items():
        if key in lower:
            return surface
    return None


def is_known_tournament(tournament_name: str) -> bool:
    return lookup_surface(tournament_name) is not None
