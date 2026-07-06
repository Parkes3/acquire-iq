# backfill_metadata.py
import pandas as pd
import os
from google_play_scraper import app
from pipeline.scrape import parse_metadata
import anthropic

OUTPUT_DIR = "data/outputs"
SAMPLE_DIR = "data/sample"

RUN_WITH_CLAUDE = True

if RUN_WITH_CLAUDE: client = anthropic.Anthropic() 
else: None

# Add all app IDs you've run — check both directories
import glob

def get_all_app_ids(directory: str) -> list[str]:
    files = glob.glob(f"{directory}/*_metadata.parquet")
    return [
        os.path.basename(f).replace("_metadata.parquet", "").replace("_", ".")
        for f in files
    ]

all_ids = get_all_app_ids(OUTPUT_DIR) + get_all_app_ids(SAMPLE_DIR)
print(f"Found {len(all_ids)} apps to backfill: {all_ids}")

for app_id in all_ids:
    try:
        result = app(app_id)
        parsed = parse_metadata(result, client=client)
        parsed["app_id"] = app_id

        # Determine which directory this app lives in
        sid = app_id.replace(".", "_")
        directory = (
            SAMPLE_DIR
            if os.path.exists(f"{SAMPLE_DIR}/{sid}_metadata.parquet")
            else OUTPUT_DIR
        )

        pd.DataFrame([parsed]).to_parquet(f"{directory}/{sid}_metadata.parquet")
        # print(f"  ✅ {app_id} — icon: {parsed.get('Icon', 'none')[:50]}...")

    except Exception as e:
        print(f"  ❌ {app_id} failed: {e}")