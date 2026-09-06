#!/usr/bin/env python3
"""Read-only SoilNet dataset inventory, duplicate audit, and stable split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import load_yaml, resolve_paths, sha256_file, sha256_text, write_csv, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FALLBACK_NAMES = ["IMG_0683.JPG", "IMG_0703.JPG", "IMG_0746.JPG", "IMG_0753.JPG", "IMG_0755.JPG"]


class DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value: int, index: int) -> None:
        node = (value, index, {})
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = (value ^ current[0]).bit_count()
            child = current[2].get(distance)
            if child is None:
                current[2][distance] = node
                return
            current = child

    def query(self, value: int, threshold: int):
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                found.append((distance, node[1]))
            low, high = distance - threshold, distance + threshold
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


def difference_hash(image: Image.Image) -> str:
    gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{value:016x}"


def normalize_legacy_path(raw: str, data_root: Path) -> str:
    value = (raw or "").strip().replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        parts = list(PureWindowsPath(raw).parts)[1:]
        value = "/".join(part.strip("\\/") for part in parts)
    value = value.lstrip("/")
    root_name = data_root.name.casefold()
    components = value.split("/")
    if components and components[0].casefold() == root_name:
        components = components[1:]
    return "/".join(components)


def inventory_images(data_root: Path):
    image_paths = sorted(
        (path for path in data_root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )
    rows = []
    for index, path in enumerate(image_paths, 1):
        relative = path.relative_to(data_root).as_posix()
        readable, width, height, perceptual_hash, error = False, "", "", "", ""
        try:
            with Image.open(path) as image:
                width, height = image.size
                perceptual_hash = difference_hash(image)
                image.load()
            readable = True
        except Exception as exc:  # audit must retain corrupt rows
            error = f"{type(exc).__name__}: {exc}"
        stat = path.stat()
        rows.append({
            "relative_path": relative,
            "filename": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "sha256": sha256_file(path),
            "image_readable": readable,
            "width": width,
            "height": height,
            "perceptual_hash": perceptual_hash,
            "duplicate_sha256_group": "",
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "read_error": error,
        })
        if index % 500 == 0:
            print(f"inventoried {index}/{len(image_paths)} images", flush=True)
    sha_groups = defaultdict(list)
    for index, row in enumerate(rows):
        sha_groups[row["sha256"]].append(index)
    for digest, indices in sha_groups.items():
        if len(indices) > 1:
            for index in indices:
                rows[index]["duplicate_sha256_group"] = f"sha256:{digest[:16]}"
    return rows


def build_duplicate_groups(rows, threshold: int):
    readable = [(index, row) for index, row in enumerate(rows) if row["perceptual_hash"]]
    dsu, tree = DSU(len(rows)), BKTree()
    candidate_pairs = 0
    for count, (index, row) in enumerate(readable, 1):
        value = int(row["perceptual_hash"], 16)
        matches = tree.query(value, threshold)
        candidate_pairs += len(matches)
        for _, other in matches:
            dsu.union(index, other)
        tree.add(value, index)
        if count % 1000 == 0:
            print(f"near-duplicate index {count}/{len(readable)}", flush=True)
    groups = defaultdict(list)
    for index, _ in readable:
        groups[dsu.find(index)].append(index)
    duplicate_rows, group_by_path = [], {}
    duplicate_group_count = 0
    for indices in sorted(groups.values(), key=lambda values: rows[min(values)]["relative_path"]):
        group_digest = sha256_text("\n".join(sorted(rows[index]["relative_path"] for index in indices)))[:16]
        group_id = f"phash:{group_digest}"
        for index in indices:
            group_by_path[rows[index]["relative_path"]] = group_id
        if len(indices) < 2:
            continue
        duplicate_group_count += 1
        hashes = {rows[index]["sha256"] for index in indices}
        relation = "EXACT" if len(hashes) == 1 else "NEAR_OR_EXACT_COMPONENT"
        for index in indices:
            row = rows[index]
            duplicate_rows.append({
                "duplicate_group_id": group_id,
                "relation": relation,
                "group_size": len(indices),
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "perceptual_hash": row["perceptual_hash"],
                "threshold": threshold,
            })
    return duplicate_rows, group_by_path, candidate_pairs, duplicate_group_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    data_root = paths["data_root"]
    audit = load_yaml(REPO / "config" / "audit.yaml")
    inventory_path = REPO / "data" / "manifests" / "dataset_inventory.csv"
    if inventory_path.exists() and not args.refresh:
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            inventory = list(csv.DictReader(handle))
        print(f"using cached inventory with {len(inventory)} images")
    else:
        inventory = inventory_images(data_root)
        write_csv(inventory_path, inventory, [
            "relative_path", "filename", "extension", "size_bytes", "sha256", "image_readable",
            "width", "height", "perceptual_hash", "duplicate_sha256_group", "mtime", "read_error",
        ])

    duplicate_rows, group_by_path, candidate_pairs, duplicate_group_count = build_duplicate_groups(
        inventory, int(audit["near_duplicate_hamming_threshold"])
    )
    write_csv(REPO / "results" / "audit" / "duplicate_report.csv", duplicate_rows, [
        "duplicate_group_id", "relation", "group_size", "relative_path", "sha256", "perceptual_hash", "threshold",
    ])

    labels_paths = sorted(data_root.rglob("labels.csv"))
    if not labels_paths:
        raise FileNotFoundError("labels.csv was not found recursively")
    labels_path = labels_paths[0]
    legacy_hash_paths = defaultdict(list)
    legacy_inventory_path = REPO / "results" / "audit" / "legacy_file_inventory.csv"
    if legacy_inventory_path.exists():
        with legacy_inventory_path.open(newline="", encoding="utf-8") as handle:
            for legacy_row in csv.DictReader(handle):
                legacy_hash_paths[legacy_row["sha256"]].append(legacy_row["relative_path"])
    metadata_rows = []
    for metadata_path in sorted(
        (path for path in data_root.rglob("*") if path.is_file() and path.suffix.casefold() in {".csv", ".xlsx"}),
        key=lambda path: path.as_posix().casefold(),
    ):
        digest = sha256_file(metadata_path)
        metadata_rows.append({
            "relative_path": metadata_path.relative_to(data_root).as_posix(),
            "file_type": metadata_path.suffix.casefold().lstrip("."),
            "size_bytes": metadata_path.stat().st_size,
            "sha256": digest,
            "role": "LABELS" if metadata_path.name.casefold() == "labels.csv" else "HISTORICAL_RESULT_OR_METADATA",
            "matching_legacy_artifact": ";".join(legacy_hash_paths.get(digest, [])),
        })
    write_csv(REPO / "results" / "audit" / "dataset_metadata_inventory.csv", metadata_rows, [
        "relative_path", "file_type", "size_bytes", "sha256", "role", "matching_legacy_artifact",
    ])
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        labels = list(csv.DictReader(handle))
    required = {"path", "SM_0", "SM_20", "light_value", "moisture_class"}
    missing_columns = sorted(required - set(labels[0])) if labels else sorted(required)

    inventory_by_relative = {row["relative_path"].casefold(): row for row in inventory}
    fallback_files = {}
    fallback_rows = []
    for original in FALLBACK_NAMES:
        fallback_name = f"aug_{original}"
        candidates = [row for row in inventory if row["filename"].casefold() == fallback_name.casefold()]
        fallback = candidates[0] if candidates else None
        if fallback:
            fallback_files[original.casefold()] = fallback
        fallback_rows.append({
            "original_filename": original,
            "fallback_filename": fallback_name,
            "exists": bool(fallback),
            "absolute_local_location": str(data_root / fallback["relative_path"]) if fallback else "MISSING",
            "relative_path": fallback["relative_path"] if fallback else "MISSING",
            "sha256": fallback["sha256"] if fallback else "MISSING",
            "width": fallback["width"] if fallback else "",
            "height": fallback["height"] if fallback else "",
            "reproducibility_status": "NOT_BITWISE_REPRODUCIBLE",
        })
    write_csv(REPO / "data" / "manifests" / "historical_fallback_mapping.csv", fallback_rows, [
        "original_filename", "fallback_filename", "exists", "absolute_local_location", "relative_path",
        "sha256", "width", "height", "reproducibility_status",
    ])

    samples = []
    for row in labels:
        relative = normalize_legacy_path(row.get("path", ""), data_root)
        record = inventory_by_relative.get(relative.casefold())
        effective = relative
        status = "PRESENT" if record and str(record["image_readable"]).casefold() == "true" else "MISSING_OR_CORRUPT"
        fallback_used = False
        if status != "PRESENT":
            fallback = fallback_files.get(Path(relative).name.casefold())
            if fallback:
                effective, record, status, fallback_used = fallback["relative_path"], fallback, "HISTORICAL_FALLBACK", True
        # Filenames/legacy IDs repeat across folders; normalized relative paths are unique.
        stable_id = relative
        samples.append({
            "sample_id": sha256_text(stable_id)[:20],
            "source_id": stable_id,
            "relative_path": relative,
            "effective_relative_path": effective,
            "path_status": status,
            "historical_fallback_used": fallback_used,
            "sha256": record["sha256"] if record else "",
            "SM_0": row.get("SM_0", ""),
            "SM_20": row.get("SM_20", ""),
            "light_value": row.get("light_value", ""),
            "moisture_class": row.get("moisture_class", ""),
            "moisture_bin": row.get("moisture_bin", ""),
            "light_bin": row.get("light_bin", ""),
        })
    write_csv(REPO / "data" / "manifests" / "soilnet_samples.csv", samples, [
        "sample_id", "source_id", "relative_path", "effective_relative_path", "path_status",
        "historical_fallback_used", "sha256", "SM_0", "SM_20", "light_value", "moisture_class",
        "moisture_bin", "light_bin",
    ])

    seed = int(audit["seed"])
    ratios = audit["split"]
    split_rows = []
    for row in samples:
        if row["path_status"] not in {"PRESENT", "HISTORICAL_FALLBACK"}:
            continue
        group_id = group_by_path.get(row["effective_relative_path"], f"sample:{row['sample_id']}")
        unit = int(sha256_text(f"{seed}:{group_id}")[:16], 16) / float(16 ** 16)
        if unit < float(ratios["train"]):
            split = "train"
        elif unit < float(ratios["train"]) + float(ratios["validation"]):
            split = "validation"
        else:
            split = "test"
        split_rows.append({**row, "group_id": group_id, "split": split})
    write_csv(REPO / "data" / "splits" / "final_split.csv", split_rows, [
        "relative_path", "effective_relative_path", "sample_id", "group_id", "split", "sha256",
        "SM_0", "SM_20", "moisture_class", "light_value", "historical_fallback_used",
    ])

    sha_counts = Counter(row["sha256"] for row in inventory)
    near_component_ids = {row["duplicate_group_id"] for row in duplicate_rows if row["relation"] == "NEAR_OR_EXACT_COMPONENT"}
    distributions = {
        column: dict(sorted(Counter(row.get(column, "") for row in labels).items(), key=lambda pair: pair[0]))
        for column in ("SM_0", "SM_20", "light_value", "moisture_class")
    }
    summary = {
        "data_root": str(data_root),
        "labels_file": labels_path.relative_to(data_root).as_posix(),
        "labels_columns": list(labels[0]) if labels else [],
        "metadata_csv_xlsx_files": len(metadata_rows),
        "metadata_files_identical_to_legacy": sum(bool(row["matching_legacy_artifact"]) for row in metadata_rows),
        "missing_required_columns": missing_columns,
        "total_files": sum(1 for path in data_root.rglob("*") if path.is_file()),
        "total_images": len(inventory),
        "labeled_rows": len(labels),
        "labeled_present_original": sum(row["path_status"] == "PRESENT" for row in samples),
        "labeled_historical_fallback": sum(row["path_status"] == "HISTORICAL_FALLBACK" for row in samples),
        "labeled_missing_or_corrupt": sum(row["path_status"] == "MISSING_OR_CORRUPT" for row in samples),
        "readable_images": sum(str(row["image_readable"]).casefold() == "true" for row in inventory),
        "corrupt_images": sum(str(row["image_readable"]).casefold() != "true" for row in inventory),
        "exact_duplicate_groups": sum(count > 1 for count in sha_counts.values()),
        "exact_duplicate_files_beyond_first": sum(max(0, count - 1) for count in sha_counts.values()),
        "unique_images_by_sha256": len(sha_counts),
        "perceptual_duplicate_components": duplicate_group_count,
        "near_duplicate_components_with_nonidentical_sha256": len(near_component_ids),
        "files_in_near_duplicate_components": sum(row["relation"] == "NEAR_OR_EXACT_COMPONENT" for row in duplicate_rows),
        "near_duplicate_candidate_pairs_at_threshold": candidate_pairs,
        "near_duplicate_hamming_threshold": int(audit["near_duplicate_hamming_threshold"]),
        "historical_2057_reconciliation": "labels.csv has 2057 data rows; it is not the count of every image under DATA_ROOT.",
        "distributions": distributions,
    }
    write_json(REPO / "results" / "audit" / "dataset_summary.json", summary)
    split_groups = defaultdict(set)
    split_group_members = defaultdict(list)
    for row in split_rows:
        split_groups[row["group_id"]].add(row["split"])
        split_group_members[row["group_id"]].append(row)
    labeled_sha_counts = Counter(row["sha256"] for row in split_rows if row["sha256"])
    summary.update({
        "labeled_unique_images_by_sha256": len(labeled_sha_counts),
        "labeled_exact_duplicate_files_beyond_first": sum(max(0, count - 1) for count in labeled_sha_counts.values()),
        "labeled_near_duplicate_groups_with_nonidentical_sha256": sum(
            len(members) > 1 and len({member["sha256"] for member in members}) > 1
            for members in split_group_members.values()
        ),
    })
    write_json(REPO / "results" / "audit" / "dataset_summary.json", summary)
    write_json(REPO / "data" / "splits" / "split_metadata.json", {
        "seed": seed,
        "ratios": ratios,
        "strategy": "stable SHA256 assignment of exact/near-duplicate connected components",
        "grouping": "Exact SHA256 plus 64-bit difference-hash distance <= threshold.",
        "capture_session_limitation": "No documented session/location identifier was found in labels.csv. Filename timestamps were not converted into session IDs because that would be an unsupported assumption.",
        "groups_crossing_splits": sum(len(splits) > 1 for splits in split_groups.values()),
        "intended_use": "Prospective reruns only. This is not an independent test for checkpoints trained on the historical 2057 rows.",
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
