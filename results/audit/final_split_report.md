# Final unique split report

Status: **NOT_GENERATED — BLOCKED_DATA_CONFLICT**

- Historical labeled records: 2,057
- SHA256-unique image bytes: 1,973
- Exact duplicate groups audited: 46
- Conflicting exact duplicate groups: 46
- Conflict fields by group: {"SM_0": 46, "SM_20": 46, "light_value": 43, "moisture_bin": 38, "moisture_class": 38}
- Canonical unique manifest: not generated
- Final train/validation/test counts: unavailable
- Group leakage: not testable because the split was not generated
- Near-duplicate cross-split leakage: not testable because the split was not generated
- Distribution audit: deferred until label conflicts are resolved with external evidence

The source files were not modified. No representative was selected because byte-identical images carry conflicting SM_0, SM_20, LI and, in many groups, moisture-class labels. Resolving this by path order would silently choose scientific labels and is forbidden.
