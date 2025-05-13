#!/usr/bin/env python3
import json, glob
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data-raw"

def load_expected_ids():
    expected = {}
    for path in glob.glob(str(DATA_DIR / "game_ids_*.json")):
        season = Path(path).stem.replace("game_ids_", "")
        ids = set(json.load(open(path, "r")))
        expected[season] = ids
    return expected

def load_fetched_ids():
    # boxscore_0022401187.json → 0022401187
    files = glob.glob(str(DATA_DIR / "boxscore_*.json"))
    ids = { Path(f).stem.replace("boxscore_", "") for f in files }
    return ids

def load_skipped():
    skipped_file = DATA_DIR / "skipped_boxscores.txt"
    if skipped_file.exists():
        return { line.strip() for line in skipped_file.read_text().splitlines() if line.strip() }
    return set()

def main():
    expected = load_expected_ids()
    fetched  = load_fetched_ids()
    skipped  = load_skipped()

    total_expected = sum(len(v) for v in expected.values())
    total_fetched  = len(fetched)
    total_skipped  = len(skipped)

    print(f"Seasons found: {', '.join(expected.keys())}")
    print(f"Total expected games: {total_expected}")
    print(f"Total fetched box scores: {total_fetched}")
    print(f"Total still skipped : {total_skipped}")

    # Show per-season breakdown
    for season, ids in expected.items():
        missing = ids - fetched
        extra   = fetched & ids  # never really “extra” but for symmetry
        print(f"\n— {season} —")
        print(f"  Expected: {len(ids)} games")
        print(f"  Fetched : {len(ids & fetched)} games")
        if missing:
            print(f"  🔴 Missing {len(missing)} IDs: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
        else:
            print("  ✅ All fetched")

    if skipped:
        print("\n‼️ WARNING: There are still skipped game IDs:")
        print(sorted(skipped))
    else:
        print("\n🎉 No skipped games remain!")

if __name__ == "__main__":
    main()
