from __future__ import annotations

import csv
import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soilnet.final_sequence import DEPLOYMENT_CHECKPOINT, MODEL_SPECS, REPO, software_environment
from soilnet.io import load_yaml, sha256_file, write_csv, write_json
from soilnet.models import build_frozen_model, model_complexity


P0_SHA256 = "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379"
WARMUP_ITERATIONS = 50
TIMED_ITERATIONS = 200


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return "UNAVAILABLE"


def raspberry_pi_environment() -> dict[str, Any]:
    device_model = _read_optional(Path("/proc/device-tree/model"))
    cpu_model = "UNAVAILABLE"
    cpuinfo = _read_optional(Path("/proc/cpuinfo"))
    for line in cpuinfo.splitlines():
        if line.casefold().startswith(("model name", "hardware")) and ":" in line:
            cpu_model = line.split(":", 1)[1].strip()
            break
    available_ram = "UNAVAILABLE"
    meminfo = _read_optional(Path("/proc/meminfo"))
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            available_ram = line.split(":", 1)[1].strip()
            break
    return {
        "platform": platform.platform(), "architecture": platform.machine(),
        "device_tree_model": device_model, "cpu_model": cpu_model,
        "logical_cores": os.cpu_count(), "available_ram": available_ram,
        "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
        "pytorch": torch.__version__, "torch_threads_at_start": torch.get_num_threads(),
    }


def require_raspberry_pi_hardware(environment: dict[str, Any]) -> None:
    evidence = " ".join((environment.get("device_tree_model", ""), environment.get("cpu_model", ""))).casefold()
    if "raspberry pi" not in evidence:
        raise RuntimeError("RPI_HARDWARE_REQUIRED")


def _benchmark_setting(model: torch.nn.Module, image: torch.Tensor, light: torch.Tensor, threads: int, label: str) -> dict[str, Any]:
    torch.set_num_threads(threads)
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            model(image, light)
        durations_ms = []
        for _ in range(TIMED_ITERATIONS):
            start = time.perf_counter_ns()
            model(image, light)
            durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(durations_ms, dtype=float)
    return {
        "setting": label, "threads": threads, "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS, "mean_latency_ms": float(values.mean()),
        "std_latency_ms": float(values.std(ddof=1)), "median_latency_ms": float(np.median(values)),
        "p90_latency_ms": float(np.percentile(values, 90)), "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)), "min_latency_ms": float(values.min()),
        "max_latency_ms": float(values.max()), "throughput_images_per_second": float(1000.0 / values.mean()),
    }


def run_raspberry_pi_benchmark() -> dict[str, Any]:
    environment = raspberry_pi_environment()
    for key, value in environment.items():
        print(f"{key}: {value}")
    require_raspberry_pi_hardware(environment)
    if not DEPLOYMENT_CHECKPOINT.is_file() or sha256_file(DEPLOYMENT_CHECKPOINT) != P0_SHA256:
        raise RuntimeError("Frozen P0 deployment checkpoint is missing or has the wrong SHA256")
    config = load_yaml(REPO / MODEL_SPECS["P0"]["config"])
    start = time.perf_counter()
    model = build_frozen_model(config)
    payload = torch.load(DEPLOYMENT_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    load_seconds = time.perf_counter() - start
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    light = torch.full((1, 1), 0.5, dtype=torch.float32)
    default_threads = max(1, int(environment["logical_cores"] or torch.get_num_threads()))
    settings = [_benchmark_setting(model, image, light, default_threads, "primary_all_logical_cores")]
    if default_threads != 1:
        settings.append(_benchmark_setting(model, image, light, 1, "secondary_single_thread"))
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    complexity = model_complexity(model, DEPLOYMENT_CHECKPOINT)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "hardware_verified": "Raspberry Pi",
        "experiment_id": MODEL_SPECS["P0"]["experiment_id"], "checkpoint_sha256": P0_SHA256,
        "checkpoint_path": str(DEPLOYMENT_CHECKPOINT.relative_to(REPO)),
        "model_loading_time_seconds": load_seconds, "input_image_shape": [1, 3, 224, 224],
        "input_LI_shape": [1, 1], "device": "cpu", "model_eval": True,
        "torch_inference_mode": True, "peak_process_rss_kib": peak_rss_kib,
        "peak_memory_scope": "Linux process maximum resident set for notebook process through benchmark completion",
        "environment": environment, "software": software_environment(), "complexity": complexity,
        "settings": settings,
    }
    output = REPO / "results/edge"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "rpi_benchmark.json", result)
    rows = [{
        **setting, "experiment_id": result["experiment_id"], "checkpoint_sha256": P0_SHA256,
        "parameters": complexity["parameters"], "trainable_parameters": complexity["trainable_parameters"],
        "checkpoint_size_bytes": complexity["checkpoint_size_bytes"], "model_loading_time_seconds": load_seconds,
        "peak_process_rss_kib": peak_rss_kib,
    } for setting in settings]
    write_csv(output / "rpi_benchmark.csv", rows, list(rows[0]))
    write_csv(output / "paper_ready_latency_summary.csv", [rows[0]], list(rows[0]))
    (output / "rpi_environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8"
    )
    primary = settings[0]
    (output / "RPI_BENCHMARK_REPORT.md").write_text(
        "# Raspberry Pi benchmark\n\n"
        f"Hardware: {environment['device_tree_model']}  \n"
        f"Frozen checkpoint SHA256: `{P0_SHA256}`  \n"
        f"Primary setting: {primary['threads']} PyTorch threads, {WARMUP_ITERATIONS} warmups, {TIMED_ITERATIONS} timed iterations.  \n"
        f"Mean latency: {primary['mean_latency_ms']:.3f} ms; median: {primary['median_latency_ms']:.3f} ms; "
        f"P95: {primary['p95_latency_ms']:.3f} ms; throughput: {primary['throughput_images_per_second']:.3f} images/s.\n",
        encoding="utf-8",
    )
    return result
