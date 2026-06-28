"""
test_matchups.py — Run a batch of well-known matchups through predict_match()
and print a clean comparison table, so you can sanity-check the model's
outputs all at once instead of running predict.py 15 separate times.

Usage:
    python test_matchups.py

Edit the MATCHUPS list below to add/remove/adjust matchups, surfaces, or
tournament context.
"""

from predict import predict_match

# Each entry: (player_a, player_b, surface, series, round_, best_of)
# series/round_/best_of are optional - use None to omit them.
MATCHUPS = [
    ("Sinner J.", "Alcaraz C.", "Hard", "Masters 1000", "Semifinals", 3),
    ("Sinner J.", "Alcaraz C.", "Clay", "Grand Slam", "The Final", 5),
    ("Djokovic N.", "Zverev A.", "Hard", None, None, 3),
    ("Djokovic N.", "Alcaraz C.", "Grass", "Grand Slam", "Semifinals", 5),
    ("Medvedev D.", "Fritz T.", "Hard", None, None, 3),
    ("Alcaraz C.", "Sinner J.", "Hard", None, None, 3),  # reverse-order check
    ("Tiafoe F.", "Rune H.", "Hard", "ATP500", "Quarterfinals", 3),
    ("Rublev A.", "Tsitsipas S.", "Clay", "Masters 1000", "3rd Round", 3),
    ("Ruud C.", "Hurkacz H.", "Clay", "Grand Slam", "2nd Round", 5),
    ("De Minaur A.", "Paul T.", "Hard", "ATP250", "1st Round", 3),
    ("Dimitrov G.", "Shelton B.", "Hard", None, None, 3),
    ("Korda S.", "Humbert U.", "Grass", "Grand Slam", "1st Round", 5),
    ("Khachanov K.", "Davidovich Fokina A.", "Clay", None, None, 3),
    ("Sinner J.", "Djokovic N.", "Hard", "Grand Slam", "Semifinals", 5),
    ("Alcaraz C.", "Zverev A.", "Clay", "Grand Slam", "The Final", 5),
]


def run_all(matchups):
    rows = []
    for player_a, player_b, surface, series, round_, best_of in matchups:
        try:
            result = predict_match(
                player_a, player_b, surface,
                series=series, round_=round_, best_of=best_of,
            )
            rows.append({
                "matchup": f"{player_a} vs {player_b}",
                "surface": surface,
                "context": f"{series or '-'} / {round_ or '-'}",
                "prob_a": f"{result['prob_a_wins']:.1%}",
                "prob_b": f"{result['prob_b_wins']:.1%}",
                "predicted_winner": result["predicted_winner"] or "(too close)",
                "confidence": result["confidence"],
                "notes": "; ".join(result["notes"]) if result["notes"] else "",
            })
        except ValueError as e:
            rows.append({
                "matchup": f"{player_a} vs {player_b}",
                "surface": surface,
                "context": f"{series or '-'} / {round_ or '-'}",
                "prob_a": "ERROR",
                "prob_b": "ERROR",
                "predicted_winner": "-",
                "confidence": "-",
                "notes": str(e),
            })
    return rows


def print_table(rows):
    headers = ["matchup", "surface", "context", "prob_a", "prob_b",
               "predicted_winner", "confidence", "notes"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}

    def fmt_row(values):
        return " | ".join(str(v).ljust(widths[h]) for h, v in zip(headers, values))

    print(fmt_row(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(fmt_row([r[h] for h in headers]))


if __name__ == "__main__":
    results = run_all(MATCHUPS)
    print_table(results)

    print()
    print("=" * 70)
    print("THINGS TO EYEBALL:")
    print("=" * 70)
    print("1. Do favorites match general tennis knowledge (e.g. top-5 player")
    print("   favored over a player ranked 30+, all else equal)?")
    print("2. Does the same player win more often on their better surface")
    print("   (e.g. a clay specialist favored more on Clay than on Grass)?")
    print("3. Row 6 (Alcaraz C. vs Sinner J.) is the reverse of row 1")
    print("   (Sinner J. vs Alcaraz C.) - probabilities should be roughly")
    print("   complementary (prob_a here ~= prob_b there), though Random")
    print("   Forest isn't perfectly symmetric so small differences are normal.")
    print("4. Any 'notes' about missing players or zero head-to-head history")
    print("   - those predictions lean more on rank/Elo defaults and are less")
    print("   reliable than ones with rich history.")
    print("5. Any ERROR rows mean a name was ambiguous - check the note for")
    print("   which players matched and use the more specific name.")
