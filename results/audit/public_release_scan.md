# Public release scan

Status: **REVIEW_REQUIRED**

- Candidate files scanned: 111
- Findings: 6
- Blocking findings: 0
- `.pth` files in repository candidate set: 0
- `.pth` rows in image manifest: 0
- Redundant checkpoint files within DATA_ROOT scope: 1
- DATA_ROOT matches to CHECKPOINT_ROOT / legacy GitHub: 0 / 4

Local absolute paths in audit evidence are reported for review; no credential value is reproduced here. Legacy-path inventories may be retained as provenance, while machine-specific runtime paths should be sanitized before public commit.

| File | Category | Severity | Action |
|---|---|---|---|
| `data/manifests/historical_fallback_mapping.csv` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |
| `results/audit/data_root_checkpoint_inventory.csv` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |
| `results/audit/dataset_summary.json` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |
| `results/audit/environment.json` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |
| `results/audit/legacy_path_references.csv` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |
| `results/audit/system_info.txt` | LOCAL_OR_LEGACY_PATH | REVIEW | path/username appears; evidence file may need sanitization or explicit retention |

Raw dataset and all 108 DATA_ROOT checkpoint binaries remain outside Git. No push was performed.
