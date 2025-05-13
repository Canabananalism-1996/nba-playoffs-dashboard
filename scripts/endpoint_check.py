# endpoint_check.py
from nba_api.stats.endpoints import boxscoretraditionalv2
import time

# your 34 missing Game-IDs
missing = [
    "0022401161",
"0022401167",
"0022401169",
"0022401170",
"0022401171",
"0022401172",
"0022401173",
"0022401174",
"0022401175",
"0022401176",
"0022401177",
"0022401178",
"0022401179",
"0022401180",
"0022401181",
"0022401182",
"0022401183",
"0022401184",
"0022401185",
"0022401186",
"0022401187",
"0022401188",
"0022401189",
"0022401190",
"0022401191",
"0022401192",
"0022401193",
"0022401194",
"0022401195",
"0022401196",
"0022401197",
"0022401198",
"0022401199",
"0022401200"
]

for gid in missing:
    try:
        # this does a GET against stats.nba.com with all the right headers
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
        df = box.get_data_frames()[0]   # player stats DataFrame
        if not df.empty:
            print(f"{gid}: ✅ {len(df)} players returned")
        else:
            print(f"{gid}: ⚠️ endpoint exists but returned zero rows")
    except Exception as e:
        # if NBA never published anything, you'll typically get an HTTPError or KeyError
        print(f"{gid}: ❌ no data ({e.__class__.__name__})")
    time.sleep(0.6)  # be kind to their rate-limits
