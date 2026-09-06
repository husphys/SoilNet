#!/usr/bin/env python3
"""Static audit of the pinned legacy checkout; never modifies legacy files."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import resolve_paths, sha256_file, sha256_text, write_csv, write_json

PINNED_COMMIT = "312c0a44605de87d31b6c52627cd91b07a4d1f75"
INVENTORY_SUFFIXES = {".ipynb", ".py", ".csv", ".xlsx", ".pth"}
WINDOWS_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/][^\"'\n\r]+)")
COLAB_PATH = re.compile(r"(?:/content/(?:drive|working)?[^\"'\s\n\r]*)")
ARTIFACT_NAME = re.compile(r"(?i)([\w .()=+\-\\/]+\.(?:pth|csv|xlsx|jpg|jpeg|png))")


def source_units(path: Path):
    if path.suffix.casefold() == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                yield f"cell:{index}", "".join(cell.get("source", []))
    elif path.suffix.casefold() == ".py":
        yield "source", path.read_text(encoding="utf-8", errors="replace")


def suspected_experiment(name: str) -> str:
    lower = name.casefold()
    mu = re.search(r"mu[=_ -]?(\d+(?:\.0)?)", lower)
    flavor = "TINYIMAGENET" if "tiny" in lower or "imagenet" in lower else "ORIGINAL"
    if "finetune" in lower or "fine_tune" in lower:
        stage = "FINETUNE"
    elif "vicreg" in lower:
        stage = "SSL"
    elif "pretrain" in lower:
        stage = "PRETRAIN"
    elif "predict" in lower or "telegram" in lower:
        stage = "DEPLOYMENT"
    else:
        stage = "UNMAPPED"
    return "_".join(part for part in (stage, flavor, f"MU{mu.group(1).replace('.0', '')}" if mu else "") if part)


def method_flags(text: str) -> dict[str, object]:
    lower = text.casefold()
    has_split = any(token in lower for token in ("random_split", "train_test_split", "groupkfold", "groupshufflesplit", "stratifiedgroupkfold"))
    loaders = set(re.findall(r"\b([A-Za-z_]\w*(?:loader|dataloader))\b", text))
    explicit_eval_loader = any(re.search(rf"\b(?:val|valid|test)\w*loader\b", name, re.I) for name in loaders)
    metric_code = any(token in lower for token in ("accuracy_score", "f1_score", "mean_squared_error", "mean_absolute_error"))
    same_training_loader = metric_code and not has_split and not explicit_eval_loader and any("dataloader" in name.casefold() for name in loaders)
    return {
        "has_train_val_test_split": has_split,
        "evaluation_on_training_loader_suspected": same_training_loader,
        "seed_present": bool(re.search(r"(?:manual_seed|random\.seed|np\.random\.seed|random_state\s*=)", text)),
        "random_choice": "random.choice" in text,
        "random_split": "random_split" in text,
        "train_test_split": "train_test_split" in text,
        "group_split": any(token in lower for token in ("groupkfold", "groupshufflesplit", "stratifiedgroupkfold", "group_id")),
        "validation_checkpoint_selection": bool(re.search(r"(?:best|save).{0,80}(?:val|valid)", text, re.I | re.S)),
        "normalization": "normalize(" in lower,
        "augmentation": any(token in lower for token in ("randomhorizontalflip", "randomresizedcrop", "colorjitter", "augmentation")),
        "target_scaled_div100": bool(re.search(r"(?:SM_0|SM_20)[^\n]{0,80}/\s*100", text)),
        "metric_rescaled_mul100": bool(re.search(r"(?:y_true|y_pred|humidity)[^\n]{0,80}\*\s*100", text)),
        "pretrained_true": "pretrained=true" in lower.replace(" ", ""),
        "pretrained_false": "pretrained=false" in lower.replace(" ", ""),
        "strict_false": "strict=false" in lower.replace(" ", ""),
        "evaluation_scope": "TRAINING_SET_METRICS_ONLY" if same_training_loader else ("SPLIT_PRESENT_REVIEW_REQUIRED" if has_split else "NO_FINAL_METRICS_DETECTED"),
    }


def main() -> int:
    configured_paths = resolve_paths()
    legacy = configured_paths["legacy_root"]
    head = subprocess.check_output(["git", "-C", str(legacy), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"Legacy checkout is {head}, expected {PINNED_COMMIT}")
    paths = sorted((path for path in legacy.rglob("*") if path.is_file() and path.suffix.casefold() in INVENTORY_SUFFIXES), key=lambda p: p.as_posix().casefold())
    inventory = []
    for path in paths:
        relative = path.relative_to(legacy).as_posix()
        inventory.append({
            "relative_path": relative,
            "file_type": path.suffix.casefold().lstrip("."),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_commit": head,
            "suspected_experiment": suspected_experiment(relative),
            "notes": "Pinned legacy artifact; unmodified.",
        })
    write_csv(REPO / "results" / "audit" / "legacy_file_inventory.csv", inventory, [
        "relative_path", "file_type", "size_bytes", "sha256", "git_commit", "suspected_experiment", "notes",
    ])

    path_references, method_rows, initialization = [], [], []
    code_paths = [path for path in paths if path.suffix.casefold() in {".ipynb", ".py"}]
    for path in code_paths:
        relative = path.relative_to(legacy).as_posix()
        full_text_parts = []
        for location, source in source_units(path):
            full_text_parts.append(source)
            references = []
            references.extend(("WINDOWS_ABSOLUTE", match.group(0).rstrip(" ,)")) for match in WINDOWS_PATH.finditer(source))
            references.extend(("COLAB", match.group(0).rstrip(" ,)")) for match in COLAB_PATH.finditer(source))
            references.extend(("ARTIFACT", match.group(1).strip()) for match in ARTIFACT_NAME.finditer(source))
            seen = set()
            for kind, reference in references:
                key = (kind, reference)
                if key in seen:
                    continue
                seen.add(key)
                path_references.append({
                    "relative_path": relative,
                    "location": location,
                    "reference_type": kind,
                    "reference": reference,
                })
        text = "\n".join(full_text_parts)
        flags = method_flags(text)
        method_rows.append({"relative_path": relative, **flags})
        for match in re.finditer(r"timm\.create_model\(\s*[\"']([^\"']+)[\"'][^\n)]*pretrained\s*=\s*(True|False)", text):
            model, pretrained = match.groups()
            initialization.append({
                "experiment": suspected_experiment(relative),
                "source_file": relative,
                "architecture": model,
                "component": "backbone_or_baseline",
                "initialization": "IMAGENET_PRETRAINED" if pretrained == "True" else "RANDOM",
                "external_pretraining": pretrained == "True",
                "checkpoint_loaded": bool(re.search(r"load_state_dict|torch\.load", text)),
                "strict_load": "FALSE" if "strict=False" in text.replace(" ", "") else "NOT_EXPLICIT_OR_TRUE",
                "fair_comparison": "NO" if path.name in {"SoilNet_pretrain_ImageNet.py", "Base_line_models_pretrain_ImageNet.py"} else "NOT_ESTABLISHED",
            })
    write_csv(REPO / "results" / "audit" / "legacy_path_references.csv", path_references, [
        "relative_path", "location", "reference_type", "reference",
    ])
    method_fields = [
        "relative_path", "has_train_val_test_split", "evaluation_on_training_loader_suspected", "seed_present",
        "random_choice", "random_split", "train_test_split", "group_split", "validation_checkpoint_selection",
        "normalization", "augmentation", "target_scaled_div100", "metric_rescaled_mul100", "pretrained_true",
        "pretrained_false", "strict_false", "evaluation_scope",
    ]
    write_csv(REPO / "results" / "audit" / "methodology_audit.csv", method_rows, method_fields)
    write_csv(REPO / "results" / "audit" / "initialization_comparison.csv", initialization, [
        "experiment", "source_file", "architecture", "component", "initialization", "external_pretraining",
        "checkpoint_loaded", "strict_load", "fair_comparison",
    ])

    metrics = [row["relative_path"] for row in inventory if row["file_type"] in {"csv", "xlsx"}]
    checkpoints = [row["relative_path"] for row in inventory if row["file_type"] == "pth"]
    base_ids = Counter(suspected_experiment(path.relative_to(legacy).as_posix()) for path in code_paths)
    graph = []
    for path in code_paths:
        relative = path.relative_to(legacy).as_posix()
        base_id = suspected_experiment(relative)
        experiment_id = base_id if base_ids[base_id] == 1 else f"{base_id}_{sha256_text(relative)[:8]}"
        lower = relative.casefold()
        mu_match = re.search(r"mu[=_ -]?(\d+)", lower)
        mu = mu_match.group(1) if mu_match else ""
        flavor_tiny = "tiny" in lower or "imagenet" in lower
        matching_metrics = [item for item in metrics if (not mu or re.search(rf"mu[_ =-]?{mu}(?:\D|$)", item, re.I)) and (("tiny" in item.casefold()) == flavor_tiny)]
        matching_checkpoints = [item for item in checkpoints if (not mu or re.search(rf"mu[_ =-]?{mu}(?:\.0)?(?:\D|$)", item, re.I)) and (("tiny" in item.casefold()) == flavor_tiny)]
        flags = next(row for row in method_rows if row["relative_path"] == relative)
        graph.append({
            "experiment_id": experiment_id,
            "stage": base_id.split("_")[0],
            "dataset": "TinyImageNet" if flavor_tiny and "finetune" not in lower else ("soil labeled + unlabeled" if "vicreg" in lower or "finetune" in lower else "TO_VERIFY"),
            "pretraining": "TinyImageNet" if flavor_tiny else ("ImageNet initialization" if flags["pretrained_true"] else "TO_VERIFY"),
            "checkpoint": ";".join(matching_checkpoints) or "MISSING_OR_UNMAPPED",
            "fine_tuning": relative if "finetune" in lower else "N/A",
            "evaluation": flags["evaluation_scope"],
            "metrics_file": ";".join(matching_metrics) or "MISSING_OR_UNMAPPED",
            "manuscript_table_or_figure": "MISSING_MANUSCRIPT_MAPPING",
            "source_notebook_or_script": relative,
        })
    write_csv(REPO / "results" / "audit" / "experiment_graph.csv", graph, [
        "experiment_id", "stage", "dataset", "pretraining", "checkpoint", "fine_tuning", "evaluation",
        "metrics_file", "manuscript_table_or_figure", "source_notebook_or_script",
    ])

    decisions = []
    for row in inventory:
        if row["file_type"] == "pth":
            status, reason = "KEEP_CHECKPOINT_REEVALUATE", "Hash and compatibility must be verified before selection."
        elif row["file_type"] in {"csv", "xlsx"}:
            status, reason = "KEEP_AS_TRAINING_LOG", "Legacy metrics are retained but evaluation scope must be established."
        else:
            status, reason = "KEEP_AS_LEGACY", "Immutable provenance at pinned commit."
        decisions.append({"relative_path": row["relative_path"], "status": status, "reason": reason})
    write_csv(REPO / "results" / "audit" / "artifact_decision.csv", decisions, ["relative_path", "status", "reason"])

    local_classical_models = set()
    for result_path in configured_paths["data_root"].rglob("*.csv"):
        try:
            with result_path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and "Model" in reader.fieldnames:
                    local_classical_models.update((row.get("Model") or "").strip().casefold() for row in reader)
        except OSError:
            continue

    def classical_claim(claim_id: str, display: str, aliases: set[str]):
        found = bool(local_classical_models & aliases)
        return (
            claim_id, "Results", display, "NEED_REEVALUATION" if found else "MISSING",
            (f"A local classifier result artifact contains a {display} row, but no split/embedding/checkpoint lineage or implementation was found."
             if found else f"No {display} code or result row was found in the pinned legacy artifacts or inventoried local result files."),
            ("Recompute cheaply from verified frozen embeddings on the prospective split."
             if found else "Provide the missing artifact or recompute cheaply from verified frozen embeddings."),
        )

    claims = [
        ("ARCH", "Methods", "SoilNet architecture", "SUPPORTED", "Legacy architecture code is present at the pinned commit.", "Document canonical architecture."),
        ("LIGHT", "Methods", "Light-intensity input", "SUPPORTED", "Legacy dual-head code concatenates a learned light feature.", "Keep input scaling explicit."),
        ("SM0", "Results", "SM_0 regression performance", "SUPPORTED_TRAINING_ONLY", "Legacy fine-tune notebooks use the training dataloader for final metrics.", "Reevaluate on truly unseen data or rerun selected model with group split."),
        ("SM20", "Results", "SM_20 regression performance", "SUPPORTED_TRAINING_ONLY", "Legacy fine-tune notebooks use the training dataloader for final metrics.", "Reevaluate on truly unseen data or rerun selected model with group split."),
        ("CLASS", "Results", "Moisture classification performance", "SUPPORTED_TRAINING_ONLY", "Accuracy/F1 were computed on the training dataloader in current fine-tune notebooks.", "Reevaluate on held-out groups."),
        ("SSL", "Methods/Results", "SSL-VICReg", "PARTIALLY_SUPPORTED", "Multiple notebooks, training CSVs, and checkpoints exist; held-out downstream evidence is unresolved.", "Verify checkpoint lineage and reevaluate selected checkpoint."),
        ("MU", "Results", "mu ablation", "SUPPORTED_TRAINING_ONLY", "Artifacts exist for multiple mu values, but stored downstream metrics are training-scope.", "Do not rerun sweep unless manuscript mapping requires it."),
        ("TINY", "Methods/Results", "TinyImageNet pretraining", "PARTIALLY_SUPPORTED", "Scripts/checkpoints exist; initialization differs from baselines and result scope needs review.", "Verify hashes and protocol."),
        ("BASE", "Results", "Baseline models", "PARTIALLY_SUPPORTED", "A baseline script exists, but it uses random initialization while SoilNet uses pretrained=True.", "Do not claim a fair comparison without matched initialization."),
        classical_claim("KNN", "KNN", {"knn", "k-nearest neighbors", "k nearest neighbors"}),
        classical_claim("GBM", "GBM", {"gbm", "gradient boosting", "gradientboosting"}),
        classical_claim("SVM", "SVM", {"svm", "svc"}),
        classical_claim("DT", "Decision Tree", {"decision tree", "decisiontree"}),
        classical_claim("MLP", "MLP", {"mlp", "multilayer perceptron"}),
        ("EDGE", "Deployment", "Edge/deployment", "PARTIALLY_SUPPORTED", "A Telegram inference notebook exists; benchmark and device evidence are not mapped.", "Provide deployment benchmark artifacts."),
        ("IRR", "Results", "Irrigation experiment", "MISSING", "No mapped artifact in the pinned legacy inventory.", "Provide artifact or remove/qualify claim."),
    ]
    claim_rows = []
    graph_by_stage = defaultdict(list)
    for row in graph:
        graph_by_stage[row["stage"]].append(row)
    for claim_id, section, claim, status, notes, action in claims:
        claim_rows.append({
            "claim_id": claim_id,
            "manuscript_section": section,
            "claim": claim,
            "claimed_value": "NOT_PROVIDED",
            "experiment_id": "UNMAPPED" if claim_id in {"KNN", "GBM", "SVM", "DT", "MLP", "IRR"} else "SEE_EXPERIMENT_GRAPH",
            "dataset_manifest": "data/manifests/soilnet_samples.csv",
            "split_manifest": "data/splits/final_split.csv (prospective only)",
            "notebook": "See results/audit/experiment_graph.csv",
            "script": "See legacy_file_inventory.csv",
            "checkpoint": "See checkpoint_inventory.csv",
            "metrics_file": "See experiment_graph.csv",
            "figure": "MISSING_MANUSCRIPT_MAPPING",
            "evidence_status": status,
            "notes": notes,
            "required_action": action,
        })
    write_csv(REPO / "results" / "audit" / "claims_evidence_matrix.csv", claim_rows, [
        "claim_id", "manuscript_section", "claim", "claimed_value", "experiment_id", "dataset_manifest",
        "split_manifest", "notebook", "script", "checkpoint", "metrics_file", "figure", "evidence_status",
        "notes", "required_action",
    ])
    write_json(REPO / "results" / "audit" / "legacy_summary.json", {
        "legacy_commit": head,
        "inventory_count": len(inventory),
        "counts_by_type": dict(Counter(row["file_type"] for row in inventory)),
        "notebooks_or_scripts": len(code_paths),
        "training_set_metrics_only_files": sum(row["evaluation_scope"] == "TRAINING_SET_METRICS_ONLY" for row in method_rows),
        "files_with_seed": sum(bool(row["seed_present"]) for row in method_rows),
        "files_with_random_choice": sum(bool(row["random_choice"]) for row in method_rows),
        "path_reference_count": len(path_references),
    })
    print(f"Audited {len(inventory)} legacy artifacts at {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
