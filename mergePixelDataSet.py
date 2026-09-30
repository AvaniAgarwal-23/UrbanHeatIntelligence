import pandas as pd
from pathlib import Path
import re

### Configuration & Paths ###
SOURCE_DIR = Path("UHI_pixel_Dataset")
OUTPUT_DIR = Path("pre-transformed datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLIT_GROUPS = {
    "training1": range(2016, 2020),
    "training2": range(2020, 2024),
    "validating": [2024],
    "final_testing": [2025]
}

FILENAME_PATTERN = re.compile(r"^UHI_Grid_(.+)_(\d{4})\.csv$", re.IGNORECASE)


### Index Source Files by Year ###
files_by_year = {}

for path in SOURCE_DIR.glob("UHI_Grid_*.csv"):
    match = FILENAME_PATTERN.match(path.name)
    if match:
        year = int(match.group(2))
        files_by_year.setdefault(year, []).append((path, match.group(1), year))


### Process and Merge Splits ###
for split_name, target_years in SPLIT_GROUPS.items():
    print(f"\nProcessing partition: {split_name}")
    split_records = []

    for year in target_years:
        entries = files_by_year.get(year, [])
        if not entries:
            print(f"[WARN] No source files found for year {year}")
            continue

        for file_path, city_name, file_year in entries:
            df = pd.read_csv(file_path)

            cols_lower = {c.lower(): c for c in df.columns}

            # Standardize city metadata
            if "city" not in cols_lower:
                df["City"] = city_name

            # Standardize year metadata
            if "year" not in cols_lower:
                df["Year"] = file_year

            split_records.append(df)
            print(f"Loaded: {file_path.name} ({len(df)} rows)")

    if split_records:
        merged_df = pd.concat(split_records, ignore_index=True)
        dest_file = OUTPUT_DIR / f"{split_name}.csv"
        merged_df.to_csv(dest_file, index=False)

        # Retrieve standardized column names
        cols_lower = {c.lower(): c for c in merged_df.columns}
        y_col = cols_lower.get("year", "Year")
        c_col = cols_lower.get("city", "City")

        print(f"\nPartition written: {dest_file.name}")
        print(f"  Rows     : {len(merged_df)}")
        print(f"  Columns  : {len(merged_df.columns)}")
        print(f"  Years    : {sorted(merged_df[y_col].unique().tolist())}")
        print(f"  Cities   : {merged_df[c_col].nunique()}")
    else:
        print(f"[WARN] No records accumulated for partition {split_name}")