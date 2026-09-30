import pandas as pd
import numpy as np
from pathlib import Path


### Dataset File Paths ###


month_file = Path(r"UHI_MontlyDataset\UHI_24_Cities_2016.csv")
grid_file  = Path(r"UHI_pixel_Dataset\UHI_Grid_Delhi_2016.csv")


### General Dataset Verification Function ###


def verify_csv(file_path, dataset_type):
    print("=" * 60)
    print(f"VERIFYING: {dataset_type}")
    print("=" * 60)

    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at '{path}'")
        return None

    print(f"Reading file: {path.name}...")
    df = pd.read_csv(path)
    print("Loaded successfully.\n")

    # Basic file metadata

    print("[File Information]")
    print(f"Filename : {path.name}")
    print(f"Rows     : {df.shape[0]}")
    print(f"Columns  : {df.shape[1]}")
    print(f"Size     : {round(path.stat().st_size / 1024, 2)} KB\n")

    # Column listing

    print("[Column Names]")
    print(df.columns.tolist(), "\n")

    # Sample records

    print("[First 5 Rows]")
    print(df.head(), "\n")

    # Data types

    print("[Data Types]")
    print(df.dtypes, "\n")

    # Missing values check

    print("[Missing Values]")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found", "\n")

    # Duplicate records check

    print("[Duplicate Rows]")
    print(f"Count: {df.duplicated().sum()}\n")

    # Statistical summary

    print("[Numeric Summary]")
    print(df.describe(), "\n")

    # Sanity checks

    print("[Basic Sanity Checks]")
    print(f"Is empty                : {df.empty}")
    print(f"All rows are duplicated : {df.duplicated().all()}\n")

    return df


### Monthly-Level Dataset Verification ###


month_df = verify_csv(month_file, "MONTH-LEVEL DATASET")

if month_df is not None:
    print("\n" + "-" * 50)
    print("MONTH-LEVEL SPECIFIC CHECKS")
    print("-" * 50)

    print("\nColumns available:")
    print(month_df.columns.tolist())

    col_lookup = {c.lower(): c for c in month_df.columns}

    # Inspect temporal and target features

    for key in ["city", "year", "month", "lst", "uhi", "uhi_binary"]:
        if key in col_lookup:
            actual_col = col_lookup[key]
            print(f"\n{actual_col}:")
            print("Unique values:", month_df[actual_col].nunique())
            print("Sample values:", month_df[actual_col].dropna().unique()[:10])

    if "month" in col_lookup:
        month_col = col_lookup["month"]
        print("\nDistinct months found:")
        print(sorted(month_df[month_col].dropna().unique()))


### Grid-Level Dataset Verification ###


grid_df = verify_csv(grid_file, "GRID-LEVEL DATASET")

if grid_df is not None:
    print("\n" + "-" * 50)
    print("GRID-LEVEL SPECIFIC CHECKS")
    print("-" * 50)

    print("\nColumns available:")
    print(grid_df.columns.tolist())

    grid_lookup = {c.lower(): c for c in grid_df.columns}

    # Inspect critical attributes

    target_attributes = [
        "city", "year", "month", "lst", "uhi",
        "ndvi", "ndwi", "ndbi", "uhi_binary"
    ]
    for key in target_attributes:
        if key in grid_lookup:
            actual_col = grid_lookup[key]
            print(f"\n{actual_col}:")
            print("Data type    :", grid_df[actual_col].dtype)
            print("Missing count:", grid_df[actual_col].isna().sum())
            print("Unique count :", grid_df[actual_col].nunique())
            print("Sample values:", grid_df[actual_col].dropna().unique()[:10])

    # Check for spatial coordinate fields

    coord_candidates = ["longitude", "latitude", "lon", "lat", ".geo", "system:index"]
    detected_coords = [c for c in grid_df.columns if c.lower() in coord_candidates]
    print("\nDetected spatial/index columns:")
    print(detected_coords)

    # Monthly row counts

    if "month" in grid_lookup:
        m_col = grid_lookup["month"]
        print(f"\nRow distribution by month ({m_col}):")
        print(grid_df.groupby(m_col).size())

    # Target variable distribution

    lst_col = grid_lookup.get("lst")
    uhi_col = grid_lookup.get("uhi")
    if lst_col and uhi_col:
        print(f"\nSummary statistics for {lst_col} & {uhi_col}:")
        print(grid_df[[lst_col, uhi_col]].describe())


### Final Validation Check ###


def final_check(df, name):
    print(f"\n[{name}] Final Pipeline Integrity Checks")
    if df is None:
        print("FAIL - DataFrame is None")
        return

    cols_lower = {c.lower() for c in df.columns}
    checks = {
        "File is not empty": not df.empty,
        "Has columns": len(df.columns) > 0,
        "Has LST column": "lst" in cols_lower,
        "Has UHI column": "uhi" in cols_lower,
        "Has year column": "year" in cols_lower,
        "Has month column": "month" in cols_lower,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "CHECK"
        print(f"  [{status}] - {check}")


final_check(month_df, "MONTH-LEVEL")
final_check(grid_df, "GRID-LEVEL")