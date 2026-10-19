from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"

# These are the dataset's expected input fields. job_id is intentionally excluded:
# it is an identifier, not a meaningful predictive feature.
TEXT_FIELDS = [
    "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "telecommuting", "has_company_logo", "has_questions",
]
TARGET = "fraudulent"


def normalize_for_duplicate_check(value: object) -> str:
    """Normalize text only for finding exact/near-exact duplicate records."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def load_and_prepare_dataset(csv_path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download the Kaggle CSV and save it as data/raw/fake_job_postings.csv. "
            "See data/raw/README.md."
        )

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}', but found: {list(df.columns)}"
        )

    # Some copies have slightly different columns. Missing optional fields are created
    # as empty columns so the pipeline remains robust.
    for col in TEXT_FIELDS:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    original_rows = len(df)

    # Exact duplicates can appear in public mirrors. Remove duplicates before splitting
    # so copies of the same posting cannot land in both training and test sets.
    duplicate_key = df[TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
    duplicate_key = duplicate_key.map(normalize_for_duplicate_check)
    df["_duplicate_key"] = duplicate_key
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_duplicate_key"], keep="first").drop(columns=["_duplicate_key"])
    duplicates_removed = before_dedup - len(df)

    # One combined text field keeps the first baseline understandable. TF-IDF is fitted
    # inside each sklearn Pipeline, so vocabulary/IDF statistics come only from training data.
    X = pd.DataFrame(index=df.index)
    X["job_text"] = (
        df[TEXT_FIELDS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    y = df[TARGET].astype(int)

    summary = {
        "rows_before_deduplication": int(original_rows),
        "exact_duplicate_rows_removed": int(duplicates_removed),
        "rows_after_deduplication": int(len(df)),
        "legitimate_count": int((y == 0).sum()),
        "fraudulent_count": int((y == 1).sum()),
        "fraudulent_rate": float(y.mean()),
        "features_used": ["combined text from " + ", ".join(TEXT_FIELDS)],
        "identifier_excluded": "job_id",
    }
    return X.reset_index(drop=True), y.reset_index(drop=True), summary
