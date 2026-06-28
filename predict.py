"""
predict.py — Standalone tennis match prediction engine.

Loads the trained model (tennis_model.pkl), the feature column order
(tennis_feature_cols.pkl), and atp_tennis.csv, then exposes:

    predict_match(player_a, player_b, surface, series=None, round_=None, best_of=3)

which returns a dict with each player's win probability and a confidence label.

DESIGN NOTES (read before modifying):

1. tennis_scaler.pkl is loaded but NOT applied before prediction. The saved
   model is a RandomForestClassifier trained on RAW (unscaled) features — this
   was verified by inspecting the actual split thresholds inside the forest,
   which are raw-magnitude values (e.g. ~64, ~1784, ~6592), not small z-score
   values. Applying the scaler here would silently corrupt every prediction.
   The scaler is kept loaded only so it's available if a future model swap
   (e.g. to Logistic Regression) needs it - it is unused by this file's logic.

2. Every engineered feature (Elo, H2H, form, surface win rate, upset rate,
   collapse rate, Elo-rank gap) is recomputed here using the EXACT SAME
   chronological logic used during training (same K-factor, same rolling
   windows, same shift-before-rolling to avoid leakage, same defaults for
   players with no history). This file computes each player's state as of
   their most recent match in atp_tennis.csv, then uses that as the "current"
   feature value for a hypothetical new matchup.

3. feature_cols.pkl defines the EXACT column order the model expects. The
   final feature row is always built via `[row[c] for c in feature_cols]` -
   never by hand-ordering columns - so a future re-save of the model with a
   different feature set will not silently break this file.

4. KNOWN BEHAVIOR - prediction asymmetry: predict_match(A, B) and
   predict_match(B, A) will NOT produce perfectly mirrored probabilities
   (e.g. P(A beats B) + P(B beats A) may not equal exactly 1.0). This is a
   real characteristic of Random Forest, not a bug in this file - tree
   ensembles aren't guaranteed to be perfectly symmetric functions of swapped
   inputs even when every individual feature is correctly mirrored (verified
   during testing: H2H, Elo, and all other features sum/match correctly across
   the swap). Training used a 50/50 random swap specifically to REDUCE this
   asymmetry, but it does not eliminate it completely. Typical gap is a few
   percentage points; if you see a much larger gap (>15-20 points) for a
   specific matchup, that's worth a closer look, but small gaps are expected
   and not something to "fix" here.
"""

import os
import glob
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_DIR, "tennis_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "tennis_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(APP_DIR, "tennis_feature_cols.pkl")

CSV_FILENAME = "atp_tennis.csv"
CSV_SEARCH_DIRS = [
    APP_DIR,
    os.getcwd(),
    os.path.join(os.path.expanduser("~"), "Downloads"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "Documents"),
]
MANUAL_CSV_PATH = ""  # set this to an absolute path if auto-search fails


def _load_pickle(path):
    """joblib-or-pickle loader, matching how the notebooks saved these files."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _find_csv():
    if MANUAL_CSV_PATH:
        if os.path.exists(MANUAL_CSV_PATH):
            return MANUAL_CSV_PATH
        raise FileNotFoundError(f"MANUAL_CSV_PATH set but not found: {MANUAL_CSV_PATH}")

    candidates = []
    for d in CSV_SEARCH_DIRS:
        candidates.extend(glob.glob(os.path.join(d, CSV_FILENAME)))
    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{CSV_FILENAME}' in: {CSV_SEARCH_DIRS}\n"
            "Set MANUAL_CSV_PATH at the top of predict.py, or place atp_tennis.csv "
            "in the same folder as this file."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
print("[predict.py] Loading model artifacts...")
model = _load_pickle(MODEL_PATH)
scaler = _load_pickle(SCALER_PATH)  # noqa: F841 - loaded but intentionally unused, see module docstring
feature_cols = _load_pickle(FEATURE_COLS_PATH)
print(f"[predict.py] Model: {type(model).__name__}, expects {len(feature_cols)} features")

# ---------------------------------------------------------------------------
# Elo / form / H2H computation (mirrors the training notebooks exactly)
# ---------------------------------------------------------------------------
K = 32
BASE_RATING = 1500.0


def _expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def _rank_to_implied_elo(rank):
    rank = max(rank, 1)
    return 1700 - 150 * np.log(rank)


class PlayerStatsStore:
    """
    Computes and stores each player's CURRENT (as of most recent match) values
    for every engineered feature, by replaying the full match history once in
    chronological order - identical logic to the training notebooks.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self._load_and_prepare(csv_path)
        self._replay_history()

    def _load_and_prepare(self, csv_path):
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Winner", "Player_1", "Player_2"]).reset_index(drop=True)
        df = df.sort_values(["Date"], kind="stable").reset_index(drop=True)
        return df

    def _replay_history(self):
        df = self.df
        n = len(df)

        self.overall_elo = {}
        self.surface_elo = {}
        self.h2h_record = {}
        self.h2h_record_surface = {}

        # Per-player rolling histories (list of (date, won, surface, rank, opp_rank))
        # stored as simple lists we can compute rolling stats from on demand.
        self.match_history = {}  # player -> list of dicts, chronological

        for row in df.itertuples(index=False):
            p1, p2 = row.Player_1, row.Player_2
            surface = row.Surface
            winner = row.Winner
            rank1, rank2 = row.Rank_1, row.Rank_2

            r1_all = self.overall_elo.get(p1, BASE_RATING)
            r2_all = self.overall_elo.get(p2, BASE_RATING)
            r1_surf = self.surface_elo.get((p1, surface), BASE_RATING)
            r2_surf = self.surface_elo.get((p2, surface), BASE_RATING)

            score1 = 1.0 if winner == p1 else 0.0
            score2 = 1.0 - score1

            exp1_all = _expected_score(r1_all, r2_all)
            self.overall_elo[p1] = r1_all + K * (score1 - exp1_all)
            self.overall_elo[p2] = r2_all + K * (score2 - (1 - exp1_all))

            exp1_surf = _expected_score(r1_surf, r2_surf)
            self.surface_elo[(p1, surface)] = r1_surf + K * (score1 - exp1_surf)
            self.surface_elo[(p2, surface)] = r2_surf + K * (score2 - (1 - exp1_surf))

            key = tuple(sorted([p1, p2]))
            surf_key = (key, surface)

            record = self.h2h_record.get(key, {})
            record[p1] = record.get(p1, 0) + (1 if winner == p1 else 0)
            record[p2] = record.get(p2, 0) + (1 if winner == p2 else 0)
            self.h2h_record[key] = record

            surf_record = self.h2h_record_surface.get(surf_key, {})
            surf_record[p1] = surf_record.get(p1, 0) + (1 if winner == p1 else 0)
            surf_record[p2] = surf_record.get(p2, 0) + (1 if winner == p2 else 0)
            self.h2h_record_surface[surf_key] = surf_record

            for player, rank, opp_rank, won in (
                (p1, rank1, rank2, winner == p1),
                (p2, rank2, rank1, winner == p2),
            ):
                self.match_history.setdefault(player, []).append({
                    "date": row.Date,
                    "won": int(won),
                    "surface": surface,
                    "is_underdog": rank > opp_rank,
                    "is_favorite": rank < opp_rank,
                })

        # Normalize player name set for fuzzy lookup
        all_names = set(df["Player_1"]) | set(df["Player_2"])
        self.known_players = sorted(all_names)
        self._lower_to_actual = {}
        for nm in self.known_players:
            self._lower_to_actual.setdefault(nm.lower(), nm)

        # Token sets for each player, used for full-name / reversed-name matching.
        # e.g. "Alcaraz C." -> {"alcaraz", "c"} ; helps match "Carlos Alcaraz" -> {"carlos","alcaraz"}
        # by overlap on the surname token, which is the most reliable shared token.
        self._tokens_to_players = {}
        for nm in self.known_players:
            tokens = self._normalize_tokens(nm)
            for tok in tokens:
                self._tokens_to_players.setdefault(tok, set()).add(nm)

        # Index of "surname -> {initials_string: actual_name}" used by the
        # initials-based resolver below. Built directly from the dataset's
        # own "Lastname F." / "Lastname F.M." format, so it's always in sync
        # with whatever abbreviation style the data actually uses - we never
        # hardcode the format, we derive it from real entries.
        # e.g. "Cerundolo J.M." -> surname "cerundolo", initials "jm"
        #      "Cerundolo F."   -> surname "cerundolo", initials "f"
        self._surname_to_initials_map = {}
        for nm in self.known_players:
            parsed = self._parse_dataset_name(nm)
            if parsed is None:
                continue
            surname, initials = parsed
            self._surname_to_initials_map.setdefault(surname, {})[initials] = nm

        self.last_seen_date = df["Date"].max()

    @staticmethod
    def _parse_dataset_name(name):
        """
        Parses a name already in the dataset's native format into
        (surname, initials). Handles MULTI-WORD surnames correctly (e.g.
        "Davidovich Fokina A.", "Van De Zandschulp B.", "O Connell C.") by
        identifying the initials suffix from its own distinctive shape -
        one or more single-letter tokens, each followed by a period (e.g.
        "A.", "J.M.", "F.") - rather than naively splitting on the first
        space, which breaks for any surname with more than one word.

        Concretely: tokenize the full name, then walk backward from the
        end collecting tokens that look like a single capital letter
        followed by a period (allowing a compound initials token like
        "J.M." which itself is multiple letter+period pairs with no
        space). Once a token doesn't match that shape, everything from
        there backward is the surname.

        Returns (surname_lowercased_joined, initials_letters_only) or None
        if the name doesn't fit this pattern at all (defensive - real data
        is occasionally messy, e.g. missing punctuation).
        """
        import re

        tokens = name.strip().split()
        if len(tokens) < 2:
            return None

        initials_pattern = re.compile(r"^([A-Z]\.){1,4}$")  # e.g. "A." or "J.M."

        initials_tokens = []
        i = len(tokens) - 1
        while i >= 0 and initials_pattern.match(tokens[i]):
            initials_tokens.insert(0, tokens[i])
            i -= 1

        if not initials_tokens:
            return None  # last token isn't initials-shaped at all - bail out

        surname_tokens = tokens[: i + 1]
        if not surname_tokens:
            return None  # nothing left for a surname - malformed entry

        surname = " ".join(surname_tokens).lower()
        initials = "".join(initials_tokens).lower().replace(".", "")
        return surname, initials

    @staticmethod
    def _full_name_to_initials(full_name):
        """
        Given a full name, returns a list of PRIORITY GROUPS of (surname,
        initials) candidates - NOT a flat list - because the two naming
        conventions are not equally likely and must not compete as equals:

        Group 1 - Western order (given name(s) first, surname last) - e.g.
        "Juan Manuel Cerundolo" -> surname "cerundolo", initials "jm".
        Tries surname = last 1 token, then last 2 tokens, etc. This is the
        convention our actual data source (The Odds API) uses for every
        name we've verified, so it is checked FIRST and EXCLUSIVELY before
        any other convention is even considered.

        Group 2 - Surname-first order (used for e.g. many Chinese, Japanese,
        Korean, and Hungarian names) - e.g. "Wu Yibing" -> surname "wu",
        initials "y". Tries surname = first 1 token, then first 2 tokens,
        etc. This group is ONLY consulted if group 1 produces no exact
        match at all - it must never be allowed to out-compete or collide
        with a valid group-1 match (this is what previously caused the
        false "Djokovic N." vs "Novak D." ambiguity: treating both
        conventions as equally-weighted guesses let an unrelated real
        player's name falsely contest the correct Western-order match).

        Returns a list of candidate-lists: [western_order_candidates,
        surname_first_candidates]. The caller must exhaust group 1 (check
        for an exact hit) before even looking at group 2.
        """
        # Normalize apostrophes to spaces before tokenizing. Different
        # sources format names like "O'Connell" inconsistently - our own
        # dataset stores this specific case as "O Connell C." (space, no
        # apostrophe). Splitting on the apostrophe here means the surname
        # token "o'connell" becomes "o connell", matching the dataset's own
        # tokenization. This must mirror the same normalization applied to
        # dataset names in _parse_dataset_name, or the two will never align.
        normalized = full_name.replace("'", " ")
        tokens = [t for t in normalized.strip().split() if t]
        if len(tokens) < 2:
            return [[], []]

        max_surname_len = min(3, len(tokens) - 1)  # cap to avoid pathological cases

        western_candidates = []
        for surname_len in range(1, max_surname_len + 1):
            given_tokens = tokens[:-surname_len]
            surname_tokens = tokens[-surname_len:]
            if not given_tokens:
                continue
            surname = " ".join(surname_tokens).lower()
            initials = "".join(t[0].lower() for t in given_tokens if t)
            western_candidates.append((surname, initials))

        surname_first_candidates = []
        for surname_len in range(1, max_surname_len + 1):
            surname_tokens = tokens[:surname_len]
            given_tokens = tokens[surname_len:]
            if not given_tokens:
                continue
            surname = " ".join(surname_tokens).lower()
            initials = "".join(t[0].lower() for t in given_tokens if t)
            candidate = (surname, initials)
            if candidate not in western_candidates:
                surname_first_candidates.append(candidate)

        return [western_candidates, surname_first_candidates]

    @staticmethod
    def _normalize_tokens(name):
        """Lowercase, strip trailing periods/commas, split into tokens, drop
        single-letter initials (they're too ambiguous to match on alone)."""
        cleaned = name.lower().replace(".", "").replace(",", "")
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return set(tokens)

    # -- name resolution -----------------------------------------------------
    def resolve_player_name(self, name):
        """
        Resolution order:
        1. Exact match (handles the dataset's native "Lastname F." format)
        2. Case-insensitive exact match
        3. Initials-based exact conversion: "Juan Manuel Cerundolo" is
           deterministically converted to candidate (surname, initials)
           pairs - e.g. ("cerundolo", "jm") - and checked against the
           dataset's own surname->initials index. This is checked BEFORE
           fuzzy matching specifically so that "Juan Manuel Cerundolo"
           resolves to "Cerundolo J.M." and NOT to the also-real
           "Cerundolo F." (Francisco Cerundolo) - fuzzy substring/token
           matching alone can't distinguish these two real players, but
           exact initials can.
        4. Substring match (handles e.g. "Alcaraz" -> "Alcaraz C.")
        5. Token-overlap match (handles e.g. "Carlos Alcaraz" -> "Alcaraz C."
           when tier 3 doesn't apply, matching on the surname even though
           word order/format differs)
        Raises ValueError on genuine ambiguity (multiple distinct players match).
        Returns None if nothing matches at all.
        """
        if name in self.known_players:
            return name

        lower = name.lower().strip()
        if lower in self._lower_to_actual:
            return self._lower_to_actual[lower]

        # --- Tier 3: initials-based exact conversion ---
        initials_match = self._resolve_via_initials(name)
        if initials_match is not None:
            return initials_match

        substring_matches = [p for p in self.known_players if lower in p.lower()]
        if len(substring_matches) == 1:
            return substring_matches[0]
        if len(substring_matches) > 1:
            raise ValueError(
                f"'{name}' matched multiple players: {substring_matches[:8]}"
                f"{'...' if len(substring_matches) > 8 else ''}. Please be more specific."
            )

        # Token-overlap fallback - e.g. query "Carlos Alcaraz" against stored "Alcaraz C."
        query_tokens = self._normalize_tokens(name)
        if not query_tokens:
            return None

        candidate_counts = {}
        for tok in query_tokens:
            for player in self._tokens_to_players.get(tok, ()):
                candidate_counts[player] = candidate_counts.get(player, 0) + 1

        if not candidate_counts:
            return None

        # Require at least one full token to match (already guaranteed by the
        # lookup above); prefer the player(s) with the most overlapping tokens.
        max_overlap = max(candidate_counts.values())
        best_candidates = sorted(p for p, c in candidate_counts.items() if c == max_overlap)

        if len(best_candidates) == 1:
            return best_candidates[0]

        raise ValueError(
            f"'{name}' matched multiple players: {best_candidates[:8]}"
            f"{'...' if len(best_candidates) > 8 else ''}. Please be more specific "
            f"(e.g. include the first-initial as it appears in the data, like 'Garcia P.')."
        )

    def _resolve_via_initials(self, full_name):
        """
        Resolves full_name using the dataset's own surname->initials index,
        in strict priority order. The surname is the strongest identifier
        in this dataset's format, so we never let a weaker or
        differently-sourced guess compete against a stronger one from a
        higher-priority group. Concretely:

        1. Western-order candidates (surname = trailing token(s), the
           convention our real data source actually uses) - checked for an
           EXACT (surname, initials) hit first. If found, return
           immediately. This alone resolves "Novak Djokovic" -> "Djokovic
           N." correctly even if an unrelated real player named
           "Novak D." exists in the data, because that player only shows
           up under the surname-first interpretation, which we have not
           even consulted yet.
        2. Still within Western-order: if no exact hit, fall back to "only
           one player has this surname at all" (handles a missing middle
           initial, e.g. "Juan Cerundolo" missing "Manuel" - falls back to
           the dataset's only Cerundolo if there's just one; if there are
           two, as with the real Cerundolos, this correctly stays empty
           and the query proceeds to ask for more specificity below).
        3. ONLY if Western-order (both passes above) found absolutely
           nothing do we consult the surname-first group, for genuinely
           surname-first names like "Wu Yibing". Same two-pass logic
           (exact hit, then lone-surname fallback) applies within this
           group too.

        Within any single pass, multiple distinct exact hits (e.g. a true
        same-surname collision like the two real Cerundolos when initials
        themselves are ambiguous) is genuine ambiguity and raises.

        Returns None if no group/pass finds anything (falls through to
        fuzzy matching).
        """
        groups = self._full_name_to_initials(full_name)

        for group in groups:
            if not group:
                continue

            # Pass A: exact (surname, initials) hits within this group only.
            exact_hits = []
            for surname, initials in group:
                initials_map = self._surname_to_initials_map.get(surname)
                if initials_map and initials in initials_map:
                    exact_hits.append(initials_map[initials])

            unique_exact = sorted(set(exact_hits))
            if len(unique_exact) == 1:
                return unique_exact[0]
            if len(unique_exact) > 1:
                raise ValueError(
                    f"'{full_name}' matched multiple players via exact surname+initials "
                    f"conversion: {unique_exact}. Please use the dataset format directly "
                    f"(e.g. 'Cerundolo J.M.')."
                )

            # Pass B: lone-occupant-of-surname fallback, still within this
            # same group only - only reached if pass A found NO exact hits.
            lone_hits = []
            for surname, initials in group:
                initials_map = self._surname_to_initials_map.get(surname)
                if initials_map and len(initials_map) == 1:
                    lone_hits.append(next(iter(initials_map.values())))

            unique_lone = sorted(set(lone_hits))
            if len(unique_lone) == 1:
                return unique_lone[0]
            if len(unique_lone) > 1:
                raise ValueError(
                    f"'{full_name}' matched multiple players via initials conversion: "
                    f"{unique_lone}. Please use the dataset format directly "
                    f"(e.g. 'Cerundolo J.M.')."
                )

            # Nothing in this group at all - proceed to the next group
            # (surname-first), rather than mixing results across groups.

        return None

    def search_players(self, query, limit=10):
        """
        Returns up to `limit` player names that plausibly match `query`, for
        autocomplete/typeahead use in a frontend. Does not raise on ambiguity -
        always returns a list (possibly empty).
        """
        lower = query.lower().strip()
        if not lower:
            return []
        exact_prefix = [p for p in self.known_players if p.lower().startswith(lower)]
        substring = [p for p in self.known_players if lower in p.lower()]
        ordered = list(dict.fromkeys(exact_prefix + substring))  # de-dupe, keep order
        return ordered[:limit]


    # -- per-player current feature lookups ----------------------------------
    def get_overall_elo(self, player):
        return self.overall_elo.get(player, BASE_RATING)

    def get_surface_elo(self, player, surface):
        return self.surface_elo.get((player, surface), BASE_RATING)

    def get_rank_implied_elo_gap(self, player, surface, current_rank):
        actual = self.get_overall_elo(player)
        implied = _rank_to_implied_elo(current_rank)
        return actual - implied

    def get_h2h(self, player_a, player_b):
        key = tuple(sorted([player_a, player_b]))
        record = self.h2h_record.get(key, {})
        wins_a = record.get(player_a, 0)
        wins_b = record.get(player_b, 0)
        total = wins_a + wins_b
        winrate_a = 0.5 if total == 0 else wins_a / total
        return winrate_a, total

    def get_h2h_surface(self, player_a, player_b, surface):
        key = tuple(sorted([player_a, player_b]))
        surf_record = self.h2h_record_surface.get((key, surface), {})
        wins_a = surf_record.get(player_a, 0)
        wins_b = surf_record.get(player_b, 0)
        total = wins_a + wins_b
        winrate_a = 0.5 if total == 0 else wins_a / total
        return winrate_a, total

    def get_form(self, player, window):
        hist = self.match_history.get(player, [])
        if not hist:
            return 0.5
        recent = hist[-window:]
        return float(np.mean([m["won"] for m in recent]))

    def get_surface_winrate(self, player, surface):
        hist = self.match_history.get(player, [])
        surf_matches = [m for m in hist if m["surface"] == surface]
        if not surf_matches:
            return 0.5
        return float(np.mean([m["won"] for m in surf_matches]))

    def get_upset_rate(self, player):
        hist = self.match_history.get(player, [])
        underdog_matches = [m for m in hist if m["is_underdog"]]
        if not underdog_matches:
            return 0.0
        return float(np.mean([m["won"] for m in underdog_matches]))

    def get_collapse_rate(self, player):
        hist = self.match_history.get(player, [])
        favorite_matches = [m for m in hist if m["is_favorite"]]
        if not favorite_matches:
            return 0.0
        return float(np.mean([1 - m["won"] for m in favorite_matches]))


# ---------------------------------------------------------------------------
# Initialize the stats store once at import time
# ---------------------------------------------------------------------------
_csv_path = _find_csv()
print(f"[predict.py] Loading match history from: {_csv_path}")
stats_store = PlayerStatsStore(_csv_path)
print(f"[predict.py] Loaded {len(stats_store.df):,} matches, "
      f"{len(stats_store.known_players):,} distinct players, "
      f"latest match date: {stats_store.last_seen_date.date()}")


# ---------------------------------------------------------------------------
# Confidence labeling
# ---------------------------------------------------------------------------
def _confidence_label(prob_winner):
    """
    prob_winner: probability of whichever player is more likely to win
    (always >= 0.5 by construction). Returns a label string.
    """
    if 0.45 <= prob_winner <= 0.55:
        return "Too Close to Call"
    elif prob_winner >= 0.70:
        return "High"
    elif prob_winner >= 0.60:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Known Series / Round values (must match the one-hot columns in feature_cols)
# ---------------------------------------------------------------------------
_SERIES_COLS = [c for c in feature_cols if c.startswith("series_")]
_ROUND_COLS = [c for c in feature_cols if c.startswith("round_")]
_KNOWN_SERIES = [c[len("series_"):] for c in _SERIES_COLS]
_KNOWN_ROUNDS = [c[len("round_"):] for c in _ROUND_COLS]


def predict_match(player_a, player_b, surface, series=None, round_=None, best_of=3):
    """
    Predict the outcome of a hypothetical match between player_a and player_b.

    Parameters
    ----------
    player_a, player_b : str
        Player names as they appear in atp_tennis.csv. Exact match is tried
        first, then case-insensitive, then partial match.
    surface : str
        One of the surfaces present in the training data (e.g. "Hard", "Clay", "Grass").
    series : str, optional
        Tournament category (e.g. "Grand Slam", "Masters 1000", "ATP250").
        If omitted or unrecognized, no series one-hot flag is set (all zero).
    round_ : str, optional
        Match round (e.g. "Quarterfinals", "1st Round"). Same fallback as series.
    best_of : int, default 3
        3 or 5.

    Returns
    -------
    dict with keys: player_a, player_b, prob_a, prob_b, predicted_winner,
    confidence, surface, notes (list of warnings, e.g. unknown player names).
    """
    notes = []

    resolved_a = stats_store.resolve_player_name(player_a)
    resolved_b = stats_store.resolve_player_name(player_b)

    if resolved_a is None:
        notes.append(f"'{player_a}' not found in match history - using default (no-history) stats.")
        resolved_a = player_a  # use the name as-is; all lookups below default safely
    if resolved_b is None:
        notes.append(f"'{player_b}' not found in match history - using default (no-history) stats.")
        resolved_b = player_b

    if series is not None and series not in _KNOWN_SERIES:
        notes.append(f"Series '{series}' not recognized (known: {_KNOWN_SERIES}). Ignoring.")
        series = None
    if round_ is not None and round_ not in _KNOWN_ROUNDS:
        notes.append(f"Round '{round_}' not recognized (known: {_KNOWN_ROUNDS}). Ignoring.")
        round_ = None

    # NOTE: we don't have live ATP rank/points here (those change weekly and
    # aren't in our static csv beyond its last update). We approximate current
    # rank using Elo-implied strength as a fallback when not supplied. For a
    # production version, rank/points should come from the live tennis API
    # (see the FastAPI integration step) rather than this approximation.
    rank_a = _elo_to_approx_rank(stats_store.get_overall_elo(resolved_a))
    rank_b = _elo_to_approx_rank(stats_store.get_overall_elo(resolved_b))
    pts_a = max(8000 / max(rank_a, 1), 1)
    pts_b = max(8000 / max(rank_b, 1), 1)

    elo_a_overall = stats_store.get_overall_elo(resolved_a)
    elo_b_overall = stats_store.get_overall_elo(resolved_b)
    elo_a_surface = stats_store.get_surface_elo(resolved_a, surface)
    elo_b_surface = stats_store.get_surface_elo(resolved_b, surface)

    h2h_winrate_a, h2h_matches = stats_store.get_h2h(resolved_a, resolved_b)
    h2h_surf_winrate_a, h2h_surf_matches = stats_store.get_h2h_surface(resolved_a, resolved_b, surface)

    form5_a, form10_a, form20_a = (stats_store.get_form(resolved_a, w) for w in (5, 10, 20))
    form5_b, form10_b, form20_b = (stats_store.get_form(resolved_b, w) for w in (5, 10, 20))

    surf_wr_a = stats_store.get_surface_winrate(resolved_a, surface)
    surf_wr_b = stats_store.get_surface_winrate(resolved_b, surface)

    upset_rate_a = stats_store.get_upset_rate(resolved_a)
    upset_rate_b = stats_store.get_upset_rate(resolved_b)
    collapse_rate_a = stats_store.get_collapse_rate(resolved_a)
    collapse_rate_b = stats_store.get_collapse_rate(resolved_b)

    elo_rank_gap_a = stats_store.get_rank_implied_elo_gap(resolved_a, surface, rank_a)
    elo_rank_gap_b = stats_store.get_rank_implied_elo_gap(resolved_b, surface, rank_b)

    row = {c: 0 for c in feature_cols}  # default everything (e.g. one-hots) to 0
    row.update({
        "Rank_1": rank_a, "Rank_2": rank_b,
        "Pts_1": pts_a, "Pts_2": pts_b,
        "elo1_overall": elo_a_overall, "elo2_overall": elo_b_overall,
        "elo1_surface": elo_a_surface, "elo2_surface": elo_b_surface,
        "h2h_winrate_p1": h2h_winrate_a, "h2h_matches_played": h2h_matches,
        "h2h_surface_winrate_p1": h2h_surf_winrate_a, "h2h_surface_matches_played": h2h_surf_matches,
        "form5_1": form5_a, "form5_2": form5_b,
        "form10_1": form10_a, "form10_2": form10_b,
        "form20_1": form20_a, "form20_2": form20_b,
        "surface_winrate_1": surf_wr_a, "surface_winrate_2": surf_wr_b,
        "upset_rate_1": upset_rate_a, "upset_rate_2": upset_rate_b,
        "collapse_rate_1": collapse_rate_a, "collapse_rate_2": collapse_rate_b,
        "elo_rank_gap_1": elo_rank_gap_a, "elo_rank_gap_2": elo_rank_gap_b,
        "Best of": best_of,
        "rank_diff": rank_a - rank_b,
        "pts_diff": pts_a - pts_b,
        "elo_overall_diff": elo_a_overall - elo_b_overall,
        "elo_surface_diff": elo_a_surface - elo_b_surface,
        "form5_diff": form5_a - form5_b,
        "form10_diff": form10_a - form10_b,
        "form20_diff": form20_a - form20_b,
        "surf_wr_diff": surf_wr_a - surf_wr_b,
        "upset_rate_diff": upset_rate_a - upset_rate_b,
        "collapse_rate_diff": collapse_rate_a - collapse_rate_b,
        "elo_rank_gap_diff": elo_rank_gap_a - elo_rank_gap_b,
    })
    if series is not None:
        col = f"series_{series}"
        if col in row:
            row[col] = 1
    if round_ is not None:
        col = f"round_{round_}"
        if col in row:
            row[col] = 1

    X = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    proba = model.predict_proba(X)[0]  # [P(class 0 = player_a loses), P(class 1 = player_a wins)]
    prob_a = float(proba[1])
    prob_b = float(1 - prob_a)

    if prob_a >= prob_b:
        predicted_winner = player_a
        confidence = _confidence_label(prob_a)
    else:
        predicted_winner = player_b
        confidence = _confidence_label(prob_b)

    if h2h_matches == 0:
        notes.append("No prior head-to-head history between these two players.")
    if resolved_a not in stats_store.match_history:
        notes.append(f"No match history at all for '{resolved_a}' - prediction uses default baseline stats.")
    if resolved_b not in stats_store.match_history:
        notes.append(f"No match history at all for '{resolved_b}' - prediction uses default baseline stats.")

    return {
        "player_a": player_a,
        "player_b": player_b,
        "resolved_player_a": resolved_a,
        "resolved_player_b": resolved_b,
        "surface": surface,
        "series": series,
        "round": round_,
        "best_of": best_of,
        "prob_a_wins": round(prob_a, 4),
        "prob_b_wins": round(prob_b, 4),
        "predicted_winner": predicted_winner if confidence != "Too Close to Call" else None,
        "confidence": confidence,
        "notes": notes,
    }


def _elo_to_approx_rank(elo):
    """
    Inverse of the rank->Elo approximation used in training, used only as a
    fallback when live rank data isn't available. Clamped to a sane range.
    elo = 1700 - 150*ln(rank)  =>  rank = exp((1700 - elo) / 150)
    """
    rank = np.exp((1700 - elo) / 150)
    return float(np.clip(rank, 1, 2000))


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 4:
        a, b, surface = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        # Fall back to the two most recent players in the dataset for a quick demo
        recent = stats_store.df.tail(1).iloc[0]
        a, b, surface = recent["Player_1"], recent["Player_2"], recent["Surface"]
        print(f"No CLI args given - demoing with the most recent match in the data: "
              f"{a} vs {b} on {surface}")

    result = predict_match(a, b, surface)
    print()
    for k, v in result.items():
        print(f"  {k}: {v}")