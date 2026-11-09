"""Row-level validation, Pandera schema checks, and validation reporting.

The row-level checks deliberately identify *which rule* rejected each row so
invalid records can be quarantined in ``data/errors/`` with an audit trail.
Pandera is used as an additional schema-level gate on the surviving clean rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera.pandas as pa

from src.utils.config import project_path, write_json

LOGGER = logging.getLogger(__name__)

USER_REQUIRED = ["user_id", "age", "gender", "occupation", "zip"]
ITEM_REQUIRED = ["item_id", "title"]
INTERACTION_REQUIRED = ["user_id", "item_id", "rating", "timestamp"]

USER_NUMERIC = ["user_id", "age"]
ITEM_NUMERIC = ["item_id"]
INTERACTION_NUMERIC = ["user_id", "item_id", "rating", "timestamp"]


def _new_failures(df: pd.DataFrame, rules: List[str]) -> pd.DataFrame:
    return pd.DataFrame(False, index=df.index, columns=rules)


def _mark(failed: pd.DataFrame, rule: str, mask: pd.Series) -> None:
    if rule not in failed.columns:
        failed[rule] = False
    failed.loc[mask.fillna(False), rule] = True


def _numeric_working_frame(
    df: pd.DataFrame, columns: List[str], failed: pd.DataFrame
) -> Dict[str, pd.Series]:
    working: Dict[str, pd.Series] = {}
    for column in columns:
        coerced = pd.to_numeric(df[column], errors="coerce")
        bad = df[column].notna() & coerced.isna()
        _mark(failed, "incorrect_dtype", bad)
        working[column] = coerced
    return working


def _missing_checks(df: pd.DataFrame, required: List[str], failed: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    missing_any = pd.Series(False, index=df.index)
    for column in required:
        missing = df[column].isna()
        counts[column] = int(missing.sum())
        missing_any = missing_any | missing
    _mark(failed, "missing_required_value", missing_any)
    return counts


def _duplicate_checks(df: pd.DataFrame, key: List[str], failed: pd.DataFrame) -> None:
    _mark(failed, "duplicate_record", df.duplicated(subset=key, keep="first"))


def _timestamp_bounds(config: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    validation_cfg = config["validation"]
    minimum = pd.Timestamp(validation_cfg["min_timestamp"], tz="UTC")
    maximum_raw = validation_cfg.get("max_timestamp", "now")
    if str(maximum_raw).lower() == "now":
        maximum = pd.Timestamp.now(tz="UTC")
    else:
        maximum = pd.Timestamp(maximum_raw, tz="UTC")
    return minimum, maximum


def _pandera_check(df: pd.DataFrame, schema: pa.DataFrameSchema) -> Tuple[bool, str | None]:
    try:
        schema.validate(df, lazy=True)
        return True, None
    except Exception as exc:  # SchemaError and compatibility errors are reported, not fatal.
        return False, str(exc).splitlines()[0][:500]


def _user_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("int64", pa.Check.greater_than(0)),
            "age": pa.Column("int64", pa.Check.in_range(1, 120)),
            "gender": pa.Column("str", pa.Check.isin(["M", "F"])),
            "occupation": pa.Column("str"),
            "zip": pa.Column("str"),
        },
        coerce=True,
        strict=False,
    )


def _item_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "item_id": pa.Column("int64", pa.Check.greater_than(0)),
            "title": pa.Column("str"),
            "release_date": pa.Column("datetime64[ns]", nullable=True, required=False),
        },
        coerce=True,
        strict=False,
    )


def _interaction_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("int64", pa.Check.greater_than(0)),
            "item_id": pa.Column("int64", pa.Check.greater_than(0)),
            "rating": pa.Column("float64"),
            "timestamp": pa.Column("int64"),
        },
        coerce=True,
        strict=False,
    )


def _finalise(
    df: pd.DataFrame,
    failed: pd.DataFrame,
    dataset: str,
    missing_counts: Dict[str, int],
    pandera_schema: pa.DataFrameSchema,
) -> Dict[str, Any]:
    rule_columns = list(failed.columns)
    failed_rules = failed.apply(
        lambda row: ";".join(column for column in rule_columns if bool(row[column])), axis=1
    )
    valid_mask = ~failed[rule_columns].any(axis=1)

    errors = df.loc[~valid_mask].copy()
    errors["failed_rules"] = failed_rules.loc[~valid_mask]
    errors["dataset"] = dataset
    valid = df.loc[valid_mask].copy()

    # Normalise surviving rows to the declared numeric dtypes before Pandera.
    numeric_columns = {
        "users": USER_NUMERIC,
        "items": ITEM_NUMERIC,
        "interactions": INTERACTION_NUMERIC,
    }[dataset]
    for column in numeric_columns:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    if dataset == "users":
        valid["age"] = valid["age"].round().astype("int64")
        valid["user_id"] = valid["user_id"].astype("int64")
        valid["occupation"] = valid["occupation"].astype(str).str.strip()
        valid["gender"] = valid["gender"].astype(str).str.upper().str.strip()
        valid["zip"] = valid["zip"].astype(str).str.strip()
    if dataset == "items":
        valid["item_id"] = valid["item_id"].astype("int64")
        valid["release_date"] = pd.to_datetime(
            valid["release_date"], errors="coerce", format="%d-%b-%Y", utc=True
        ).dt.tz_localize(None)
    if dataset == "interactions":
        valid["user_id"] = valid["user_id"].astype("int64")
        valid["item_id"] = valid["item_id"].astype("int64")
        valid["rating"] = valid["rating"].astype("float64")
        valid["timestamp"] = valid["timestamp"].astype("int64")

    pandera_passed, pandera_error = _pandera_check(valid, pandera_schema)
    checks = []
    for rule in rule_columns:
        count = int(failed[rule].sum())
        checks.append(
            {
                "dataset": dataset,
                "check": rule,
                "count": count,
                "status": "PASS" if count == 0 else "FAIL",
            }
        )
    checks.append(
        {
            "dataset": dataset,
            "check": "pandera_schema",
            "count": 0 if pandera_passed else 1,
            "status": "PASS" if pandera_passed else "FAIL",
        }
    )
    return {
        "valid": valid,
        "errors": errors,
        "checks": checks,
        "rows_in": int(len(df)),
        "rows_valid": int(len(valid)),
        "rows_rejected": int(len(errors)),
        "missing_counts": missing_counts,
        "pandera_passed": pandera_passed,
        "pandera_error": pandera_error,
    }


def _validate_users(
    users: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    rules = [
        "missing_required_value",
        "duplicate_record",
        "incorrect_dtype",
        "invalid_age",
        "invalid_gender",
        "invalid_id",
    ]
    failed = _new_failures(users, rules)
    missing = _missing_checks(users, USER_REQUIRED, failed)
    _duplicate_checks(users, config["validation"]["duplicate_key"]["users"], failed)
    working = _numeric_working_frame(users, USER_NUMERIC, failed)
    _mark(failed, "invalid_age", working["age"].notna() & ~working["age"].between(1, 120))
    gender = users["gender"].astype("string").str.upper().str.strip()
    _mark(failed, "invalid_gender", users["gender"].notna() & ~gender.isin(["M", "F"]))
    _mark(failed, "invalid_id", working["user_id"].notna() & (working["user_id"] <= 0))
    return _finalise(users, failed, "users", missing, _user_schema())


def _validate_items(
    items: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    rules = [
        "missing_required_value",
        "duplicate_record",
        "incorrect_dtype",
        "invalid_date",
        "invalid_id",
    ]
    failed = _new_failures(items, rules)
    missing = _missing_checks(items, ITEM_REQUIRED, failed)
    _duplicate_checks(items, config["validation"]["duplicate_key"]["items"], failed)
    working = _numeric_working_frame(items, ITEM_NUMERIC, failed)
    _mark(failed, "invalid_id", working["item_id"].notna() & (working["item_id"] <= 0))
    parsed_release = pd.to_datetime(
        items["release_date"], errors="coerce", format="%d-%b-%Y", utc=True
    )
    _mark(failed, "invalid_date", items["release_date"].notna() & parsed_release.isna())
    # Release dates may legitimately predate the interaction window; only reject
    # dates that are in the future or unparseable.
    _, maximum = _timestamp_bounds(config)
    _mark(failed, "invalid_date", parsed_release.notna() & (parsed_release > maximum))
    return _finalise(items, failed, "items", missing, _item_schema())


def _validate_interactions(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    rules = [
        "missing_required_value",
        "duplicate_record",
        "incorrect_dtype",
        "invalid_rating",
        "invalid_timestamp",
        "invalid_id",
        "orphan_user_id",
        "orphan_item_id",
    ]
    failed = _new_failures(interactions, rules)
    missing = _missing_checks(interactions, INTERACTION_REQUIRED, failed)
    _duplicate_checks(
        interactions, config["validation"]["duplicate_key"]["interactions"], failed
    )
    working = _numeric_working_frame(interactions, INTERACTION_NUMERIC, failed)

    rating_min = float(config["validation"]["rating_min"])
    rating_max = float(config["validation"]["rating_max"])
    _mark(
        failed,
        "invalid_rating",
        working["rating"].notna()
        & ((working["rating"] < rating_min) | (working["rating"] > rating_max)),
    )

    parsed_ts = pd.to_datetime(working["timestamp"], unit="s", errors="coerce", utc=True)
    minimum, maximum = _timestamp_bounds(config)
    _mark(failed, "invalid_timestamp", interactions["timestamp"].notna() & parsed_ts.isna())
    _mark(
        failed,
        "invalid_timestamp",
        parsed_ts.notna() & ((parsed_ts < minimum) | (parsed_ts > maximum)),
    )
    _mark(failed, "invalid_id", working["user_id"].notna() & (working["user_id"] <= 0))
    _mark(failed, "invalid_id", working["item_id"].notna() & (working["item_id"] <= 0))

    user_ids = set(pd.to_numeric(users["user_id"], errors="coerce").dropna().astype("int64"))
    item_ids = set(pd.to_numeric(items["item_id"], errors="coerce").dropna().astype("int64"))
    _mark(
        failed,
        "orphan_user_id",
        working["user_id"].notna() & ~working["user_id"].isin(user_ids),
    )
    _mark(
        failed,
        "orphan_item_id",
        working["item_id"].notna() & ~working["item_id"].isin(item_ids),
    )
    return _finalise(interactions, failed, "interactions", missing, _interaction_schema())


def _write_validation_report(
    config: Dict[str, Any], results: Dict[str, Dict[str, Any]], faults_injected: bool
) -> Dict[str, Any]:
    reports_dir = project_path(config, "reports")
    errors_dir = project_path(config, "data_errors")
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "faults_injected": faults_injected,
        "datasets": {},
        "overall_status": "PASS",
    }
    for dataset, result in results.items():
        checks.extend(result["checks"])
        summary["datasets"][dataset] = {
            "rows_in": result["rows_in"],
            "rows_valid": result["rows_valid"],
            "rows_rejected": result["rows_rejected"],
            "missing_counts": result["missing_counts"],
            "pandera_schema_passed": result["pandera_passed"],
            "pandera_error": result["pandera_error"],
            "error_file": str(errors_dir / f"{dataset}_errors.parquet"),
        }
        if result["rows_rejected"] > 0 or not result["pandera_passed"]:
            summary["overall_status"] = "FAULTS_DETECTED"

        # Error tables intentionally preserve mixed raw values (for example a
        # string in an integer column). Parquet needs one physical type per
        # column, so object columns are stringified only in the Parquet copy.
        parquet_errors = result["errors"].copy()

        def _stringify(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, float) and pd.isna(value):
                return None
            return str(value)

        for column in parquet_errors.select_dtypes(include=["object", "str"]).columns:
            parquet_errors[column] = parquet_errors[column].map(_stringify)
        parquet_errors.to_parquet(errors_dir / f"{dataset}_errors.parquet", index=False)
        result["errors"].to_csv(errors_dir / f"{dataset}_errors.csv", index=False)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(reports_dir / "validation_report.csv", index=False)
    write_json(reports_dir / "validation_report.json", {"summary": summary, "checks": checks})

    lines = [
        "# Data Validation Report",
        "",
        f"**Fault injection enabled:** {'yes' if faults_injected else 'no'}",
        "",
        f"**Overall status:** `{summary['overall_status']}`",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Rows in | Rows valid | Rows rejected | Pandera schema |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, info in summary["datasets"].items():
        pandera_status = "PASS" if info["pandera_schema_passed"] else "FAIL"
        lines.append(
            f"| {dataset} | {info['rows_in']} | {info['rows_valid']} | "
            f"{info['rows_rejected']} | {pandera_status} |"
        )

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Dataset | Check | Rows flagged | Status |",
            "|---|---|---:|---|",
        ]
    )
    for row in checks:
        lines.append(
            f"| {row['dataset']} | {row['check']} | {row['count']} | {row['status']} |"
        )

    lines.extend(["", "## Missing-value counts", ""])
    for dataset, info in summary["datasets"].items():
        counts = ", ".join(f"{column}={count}" for column, count in info["missing_counts"].items())
        lines.append(f"- **{dataset}**: {counts or 'none'}")

    lines.extend(
        [
            "",
            "## Quarantined records",
            "",
            "Invalid rows are written to `data/errors/` as both Parquet and CSV. "
            "The `failed_rules` column records every rule that rejected the row.",
            "",
            "- `data/errors/users_errors.parquet`",
            "- `data/errors/items_errors.parquet`",
            "- `data/errors/interactions_errors.parquet`",
            "",
        ]
    )
    (reports_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def validate_data(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    config: Dict[str, Any],
    faults_injected: bool = False,
) -> Dict[str, Any]:
    """Validate all three tables, quarantine bad rows, and write reports."""
    user_result = _validate_users(users, config)
    item_result = _validate_items(items, config)
    interaction_result = _validate_interactions(interactions, users, items, config)
    results = {
        "users": user_result,
        "items": item_result,
        "interactions": interaction_result,
    }
    summary = _write_validation_report(config, results, faults_injected)

    for dataset, result in results.items():
        LOGGER.info(
            "Validated %s: in=%d valid=%d rejected=%d pandera=%s",
            dataset,
            result["rows_in"],
            result["rows_valid"],
            result["rows_rejected"],
            result["pandera_passed"],
        )
    if faults_injected:
        LOGGER.warning("Validation detected injected faults; rejected rows were quarantined.")

    return {
        "users": user_result["valid"],
        "items": item_result["valid"],
        "interactions": interaction_result["valid"],
        "errors": {name: result["errors"] for name, result in results.items()},
        "report": summary,
    }
