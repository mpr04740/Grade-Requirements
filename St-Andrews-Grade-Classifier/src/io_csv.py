import streamlit as st
import pandas as pd
from typing import List, Tuple

# ------------------------
# CSV helpers (UI-side)
# ------------------------

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # allow singular "credit"
    if "credit" in df.columns and "credits" not in df.columns:
        df = df.rename(columns={"credit": "credits"})
    return df

def read_csv_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return _normalise_cols(df)

def validate_completed_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"grade", "credits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Expected: Grade, Credits.")
    out = df[["grade", "credits"]].copy()
    out = out.rename(columns={"grade": "Grade", "credits": "Credits"})
    return out

def validate_remaining_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"credits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Expected: Credits.")
    out = df[["credits"]].copy()
    out = out.rename(columns={"credits": "Credits"})
    return out

def parse_completed(df: pd.DataFrame) -> List[Tuple[float, float]]:
    rows = []
    for _, row in df.iterrows():
        grade = row.get("Grade")
        credit = row.get("Credits")
        if pd.isna(grade) or pd.isna(credit):
            continue
        if float(credit) <= 0:
            continue
        rows.append((float(grade), float(credit)))
    return rows

def parse_outstanding(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Only credits matter for requirements; we use 0.0 as a placeholder grade.
    """
    rows = []
    for _, row in df.iterrows():
        credit = row.get("Credits")
        if pd.isna(credit):
            continue
        if float(credit) <= 0:
            continue
        rows.append((0.0, float(credit)))
    return rows
