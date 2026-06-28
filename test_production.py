"""
test_production.py — Hits your DEPLOYED backend over real HTTP (not a local
Python import like test_matchups.py does) to verify the live production
deployment actually works end to end: health check, live fixtures, a real
prediction, and the accuracy dashboard.

Usage:
    python test_production.py https://your-backend-url.onrender.com

If you omit the URL, it defaults to http://127.0.0.1:8000 (useful for
testing your local server the same way the production one gets tested,
as a sanity check that this script itself works before pointing it at Render).
"""

import sys
import json
import time

import requests


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    base_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    print(f"Testing backend at: {base_url}\n")

    all_passed = True

    # --- 1. Health check ---
    print("=" * 70)
    print("1. GET /health")
    print("=" * 70)
    try:
        t0 = time.time()
        r = requests.get(f"{base_url}/health", timeout=60)  # generous timeout - Render free tier cold starts can take 30-60s
        elapsed = time.time() - t0
        ok = check("Health endpoint responds with 200", r.status_code == 200, f"status={r.status_code}, took {elapsed:.1f}s")
        all_passed &= ok
        if ok:
            data = r.json()
            print(f"       matches_loaded: {data.get('matches_loaded')}")
            print(f"       distinct_players: {data.get('distinct_players')}")
            print(f"       latest_match_date: {data.get('latest_match_date')}")
            if elapsed > 10:
                print(f"       NOTE: took {elapsed:.1f}s - likely a cold start (Render free tier spins down when idle).")
                print(f"             Subsequent requests should be fast until it idles again.")
    except requests.RequestException as e:
        check("Health endpoint responds with 200", False, f"Could not connect: {e}")
        print("\nStopping here - if /health fails, nothing else will work.")
        print("Check: is the Render service actually running? Check the Render dashboard logs.")
        sys.exit(1)

    # --- 2. Live fixtures ---
    print("\n" + "=" * 70)
    print("2. GET /fixtures/upcoming")
    print("=" * 70)
    try:
        r = requests.get(f"{base_url}/fixtures/upcoming", timeout=30)
        ok = check("Fixtures endpoint responds with 200", r.status_code == 200, f"status={r.status_code}")
        all_passed &= ok
        fixtures = r.json() if ok else []
        if ok:
            print(f"       {len(fixtures)} live fixture(s) found")
            for f in fixtures[:5]:
                print(f"       - {f['player_a']} vs {f['player_b']} | {f['tournament']} | "
                      f"{f['surface']} (known={f['surface_known']})")
            if len(fixtures) == 0:
                print("       NOTE: zero fixtures is NOT necessarily a bug - it just means no ATP")
                print("             tournament is currently active in The Odds API's data right now.")
    except requests.RequestException as e:
        check("Fixtures endpoint responds with 200", False, str(e))
        fixtures = []

    # --- 3. A real prediction ---
    print("\n" + "=" * 70)
    print("3. POST /predict")
    print("=" * 70)
    if fixtures:
        f = fixtures[0]
        payload = {"player_a": f["player_a"], "player_b": f["player_b"], "surface": f["surface"]}
        print(f"       Using live fixture: {f['player_a']} vs {f['player_b']}")
    else:
        payload = {"player_a": "Sinner J.", "player_b": "Alcaraz C.", "surface": "Hard"}
        print("       No live fixtures available - using a fallback matchup from the dataset")

    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        ok = check("Predict endpoint responds with 200", r.status_code == 200, f"status={r.status_code}")
        all_passed &= ok
        if ok:
            result = r.json()
            print(f"       {result['resolved_player_a']} {result['prob_a_wins']:.1%} vs "
                  f"{result['prob_b_wins']:.1%} {result['resolved_player_b']}")
            print(f"       Predicted winner: {result['predicted_winner']} ({result['confidence']})")
            sane = 0.0 <= result["prob_a_wins"] <= 1.0 and abs(result["prob_a_wins"] + result["prob_b_wins"] - 1.0) < 0.01
            all_passed &= check("Probabilities are sane (sum to ~1.0, in [0,1])", sane)
        else:
            print(f"       Response: {r.text[:300]}")
    except requests.RequestException as e:
        check("Predict endpoint responds with 200", False, str(e))

    # --- 4. Dashboard ---
    print("\n" + "=" * 70)
    print("4. GET /dashboard/accuracy")
    print("=" * 70)
    try:
        r = requests.get(f"{base_url}/dashboard/accuracy", timeout=30)
        ok = check("Dashboard endpoint responds with 200", r.status_code == 200, f"status={r.status_code}")
        all_passed &= ok
        if ok:
            report = r.json()
            print(f"       total_tracked={report['total_tracked']} pending={report['pending']} "
                  f"graded={report['graded']} overall_accuracy={report['overall_accuracy']}")
    except requests.RequestException as e:
        check("Dashboard endpoint responds with 200", False, str(e))

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED - production deployment is working.")
    else:
        print("SOME CHECKS FAILED - see [FAIL] lines above for what to investigate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
