import os
import math
import re

import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator


# Config
INPUT_FILE = "/opt/airflow/data/online-harms-data-tables-adults.xlsx"
SHEET_NAME = "Data"
OUT_DIR = "/opt/airflow/output"
OUT_CSV = os.path.join(OUT_DIR, "online_harms_tidy.csv")
OUT_DICT = os.path.join(OUT_DIR, "online_harms_data_dictionary.json")

os.makedirs(OUT_DIR, exist_ok=True)


def _simplify_label(label: str | float | int | None) -> str | None:
    # Drop letters
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    s = str(label)
    s = re.sub(r"\s*\([a-zA-Z]+\)\s*$", "", s).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_qcode_text(cell: str) -> tuple[str | None, str | None]:
    # Get question code
    if not isinstance(cell, str):
        return None, None
    m = re.match(r"\s*<([^>]+)>\s*(.*)", cell)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, cell.strip()

def _build_crossbreak_map(df: pd.DataFrame, start_idx: int) -> tuple[list[str], list[str | None]]:
    # Get crossbreak
    hdr_top = df.iloc[start_idx + 3, 1:]  
    hdr_bot = df.iloc[start_idx + 4, 1:]

    names: list[str] = []
    groups: list[str | None] = []

    current = None
    for top, bot in zip(hdr_top.tolist(), hdr_bot.tolist()):
        if isinstance(top, str) and top.strip():
            current = top.strip()
        names.append(current if current else "All respondents")
        groups.append(_simplify_label(bot))
    return names, groups

def _parse_block(df: pd.DataFrame, start_idx: int) -> list[dict]:
    q_code, q_text = _parse_qcode_text(df.iloc[start_idx, 0])

    cross_names, cross_groups = _build_crossbreak_map(df, start_idx)

    # Get Unweighted, Effective, Total (weighted base)
    base_rows = {
        "unweighted": start_idx + 5,
        "effective": start_idx + 6,
        "weighted_base": start_idx + 7,
    }

    bases = {k: pd.to_numeric(df.iloc[r, 1:], errors="coerce").tolist() for k, r in base_rows.items()}

    records: list[dict] = []
    r = start_idx + 8
    n = len(df)

    while r < n:
        first = df.iloc[r, 0]
        if isinstance(first, str) and (first.startswith("<") or first.strip() == "Crossbreak"):
            break
        if isinstance(first, str) and "95% lower case" in first:
            r += 1
            continue
        if not isinstance(first, str) or not first.strip():
            r += 1
            continue

        # Skip already read rows
        if first in ("Unweighted row", "Effective sample size", "Total"):
            r += 1
            continue

        label = _simplify_label(first)
        vals = pd.to_numeric(df.iloc[r, 1:], errors="coerce").tolist()

        for i_col, v in enumerate(vals):
            wb = bases["weighted_base"][i_col]
            uw = bases["unweighted"][i_col]
            eff = bases["effective"][i_col]

            pct = np.nan
            if pd.notna(v) and pd.notna(wb) and wb not in (0, np.nan):
                pct = (float(v) / float(wb)) * 100.0

            # Write
            records.append({
                "question_code": q_code,
                "question_text": q_text,
                "response_label": label,
                "crossbreak": cross_names[i_col],
                "group": cross_groups[i_col],
                "weighted_base": float(wb) if pd.notna(wb) else None,
                "unweighted_base": float(uw) if pd.notna(uw) else None,
                "effective_sample_size": float(eff) if pd.notna(eff) else None,
                "weighted_count": float(v) if pd.notna(v) else None,
                "percent": float(pct) if pd.notna(pct) else None,
            })
        r += 1

    return records

def transform_and_write():
    # Read 
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")

    first_col = df.columns[0]
    is_question = df[first_col].astype(str).str.startswith("<", na=False)
    question_indices = df.index[is_question].tolist()

    all_rows: list[dict] = []
    for idx in question_indices:
        all_rows.extend(_parse_block(df, idx))

    out = pd.DataFrame(all_rows)

    # Clean deduplicates (just in case)
    out.drop_duplicates(
        subset=["question_code", "response_label", "crossbreak", "group"],
        keep="first",
        inplace=True
    )

    # Numeric types
    numeric_cols = ["weighted_base", "unweighted_base", "effective_sample_size", "weighted_count", "percent"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Percent error handling
    out.loc[out["percent"] > 100, "percent"] = np.nan
    out.loc[out["percent"] < 0, "percent"] = np.nan
    out.loc[out["weighted_base"].isna() | (out["weighted_base"] == 0), "percent"] = np.nan

    # Write outputs (CSV + Data Dictionary)
    out.sort_values(["question_code", "response_label", "crossbreak", "group"], inplace=True)
    out.to_csv(OUT_CSV, index=False)

    # Write metadata
    data_dict = {
        "columns": {
            "question_code": "Short code inside angle brackets for the question (e.g., 'C3a').",
            "question_text": "Full question wording.",
            "response_label": "Answer/row label within the question (e.g., 'Scams / fraud').",
            "crossbreak": "Dimension name (e.g., 'Gender', 'Age group (A)', 'Region', 'All respondents').",
            "group": "Group within the crossbreak (letters like '(a)' removed).",
            "weighted_base": "Weighted base ('Total' row) for that crossbreak group.",
            "unweighted_base": "Unweighted row for that crossbreak group.",
            "effective_sample_size": "Effective sample size for that crossbreak group.",
            "weighted_count": "Weighted count/value for this response in this group.",
            "percent": "weighted_count / weighted_base, expressed as a percentage (0–100)."
        }
    }
    pd.Series(data_dict).to_json(OUT_DICT)

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_DICT}")


# DAG

with DAG(
    dag_id="online_harms_pipeline",
    schedule=None, 
    catchup=False,
) as dag:

    transform = PythonOperator(
        task_id="transform_online_harms",
        python_callable=transform_and_write,
    )

    transform

