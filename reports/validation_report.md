# Data Validation Report

**Fault injection enabled:** yes

**Overall status:** `FAULTS_DETECTED`

## Dataset summary

| Dataset | Rows in | Rows valid | Rows rejected | Pandera schema |
|---|---:|---:|---:|---|
| users | 945 | 943 | 2 | PASS |
| items | 1684 | 1682 | 2 | PASS |
| interactions | 100001 | 99991 | 10 | PASS |

## Checks

| Dataset | Check | Rows flagged | Status |
|---|---|---:|---|
| users | missing_required_value | 1 | FAIL |
| users | duplicate_record | 1 | FAIL |
| users | incorrect_dtype | 0 | PASS |
| users | invalid_age | 0 | PASS |
| users | invalid_gender | 0 | PASS |
| users | invalid_id | 0 | PASS |
| users | pandera_schema | 0 | PASS |
| items | missing_required_value | 1 | FAIL |
| items | duplicate_record | 1 | FAIL |
| items | incorrect_dtype | 0 | PASS |
| items | invalid_date | 0 | PASS |
| items | invalid_id | 0 | PASS |
| items | pandera_schema | 0 | PASS |
| interactions | missing_required_value | 1 | FAIL |
| interactions | duplicate_record | 1 | FAIL |
| interactions | incorrect_dtype | 2 | FAIL |
| interactions | invalid_rating | 2 | FAIL |
| interactions | invalid_timestamp | 3 | FAIL |
| interactions | invalid_id | 0 | PASS |
| interactions | orphan_user_id | 1 | FAIL |
| interactions | orphan_item_id | 1 | FAIL |
| interactions | pandera_schema | 0 | PASS |

## Missing-value counts

- **users**: user_id=0, age=0, gender=1, occupation=0, zip=0
- **items**: item_id=0, title=1
- **interactions**: user_id=0, item_id=0, rating=1, timestamp=0

## Quarantined records

Invalid rows are written to `data/errors/` as both Parquet and CSV. The `failed_rules` column records every rule that rejected the row.

- `data/errors/users_errors.parquet`
- `data/errors/items_errors.parquet`
- `data/errors/interactions_errors.parquet`
