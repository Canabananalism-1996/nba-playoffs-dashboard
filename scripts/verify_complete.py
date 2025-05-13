#!/usr/bin/env python3
import json, glob
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data-raw"

# Load expected IDs
expected = 0
for fn in glob.glob(str(DATA_DIR/"game_ids_*.json")):
    expected += len(json.load(open(fn)))

# Count fetched boxscores
fetched = len(list((DATA_DIR).glob("boxscore_*.json")))

# Count skipped
skipped = sum(1 for l in open(DATA_DIR/"skipped_boxscores.txt") if l.strip())

print(f"Expected games: {expected}")
print(f"Fetched games : {fetched}")
print(f"Skipped games : {skipped}")

if fetched + skipped == expected:
    print("\n✅ All game IDs accounted for! 🎉")
else:
    print("\n⚠️ Mismatch detected: some game IDs may be missing.")
