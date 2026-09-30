import pandas as pd
from pathlib import Path

### Configuration & Paths ###
DATA_DIR = Path("UHI_MontlyDataset")
OUTPUT_DIR = Path("pre-transformed datasets")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "UHI_24_Cities_2016_2025.csv"
YEARS = range(2016, 2026)


### Ingestion & Concatenation ###
records = []

for year in YEARS:
    csv_path = DATA_DIR / f"UHI_24_Cities_{year}.csv"

    if not csv_path.exists():
        print(f"[WARN] File missing: {csv_path.name}")
        continue

    df = pd.read_csv(csv_path)

    # Standardize 'year' column if not present
    existing_cols = {c.lower(): c for c in df.columns}
    if "year" not in existing_cols:
        df["year"] = year

    records.append(df)
    print(f"Loaded {csv_path.name} ({len(df)} records)")


### Export Merged Dataset ###
if records:
    merged_df = pd.concat(records, ignore_index=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset successfully consolidated:")
    print(f"  Target  : {OUTPUT_FILE}")
    print(f"  Shape   : {merged_df.shape[0]} rows x {merged_df.shape[1]} cols")
    print(f"  Years   : {sorted(merged_df['year'].unique().tolist())}")
else:
    print("[ERROR] No datasets found to merge.")