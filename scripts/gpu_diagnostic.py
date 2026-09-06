#!/usr/bin/env python3
"""Non-mutating WSL/PyTorch GPU diagnostic; never installs drivers or CUDA."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from soilnet.io import write_json


def main() -> int:
    nvidia_smi = Path("/usr/lib/wsl/lib") / "nvidia-smi"
    command = [str(nvidia_smi)] if nvidia_smi.exists() else ["nvidia-smi"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        nvml_output = (process.stdout + process.stderr).strip()
        nvml_ok = process.returncode == 0
        nvml_exit_code = process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        nvml_output = f"{type(exc).__name__}: {exc}"
        nvml_ok = False
        nvml_exit_code = None

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = "UNAVAILABLE"
    tensor_smoke = False
    tensor_error = ""
    if cuda_available and device_count:
        try:
            device_name = torch.cuda.get_device_name(0)
            left = torch.ones((2, 2), device="cuda")
            right = torch.ones((2, 2), device="cuda")
            tensor_smoke = bool(torch.equal(left @ right, torch.full((2, 2), 2.0, device="cuda")))
        except Exception as exc:  # diagnostic must record rather than conceal runtime failure
            tensor_error = f"{type(exc).__name__}: {exc}"

    if cuda_available and tensor_smoke and nvml_ok:
        status = "CUDA_WORKING"
    elif cuda_available and tensor_smoke:
        status = "CUDA_AVAILABLE_NVML_LIMITED"
    elif not cuda_available:
        status = "CUDA_UNAVAILABLE"
    else:
        status = "NVML_UNAVAILABLE"

    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "torch_cuda_device_name_0": device_name,
        "cuda_tensor_matmul_smoke": tensor_smoke,
        "cuda_tensor_error": tensor_error,
        "nvidia_smi_command": command,
        "nvidia_smi_exists_in_wsl_lib": nvidia_smi.exists(),
        "nvidia_smi_exit_code": nvml_exit_code,
        "nvml_available": nvml_ok,
        "nvidia_smi_output": nvml_output,
    }
    write_json(REPO / "results" / "audit" / "gpu_diagnostic.json", report)
    text = [
        f"captured_utc: {report['captured_utc']}",
        f"status: {status}",
        f"torch.__version__: {torch.__version__}",
        f"torch.version.cuda: {torch.version.cuda}",
        f"torch.cuda.is_available(): {cuda_available}",
        f"torch.cuda.device_count(): {device_count}",
        f"torch.cuda.get_device_name(0): {device_name}",
        f"CUDA tensor/matmul smoke: {tensor_smoke}",
        f"nvidia-smi command: {' '.join(command)}",
        f"nvidia-smi exit code: {nvml_exit_code}",
        "nvidia-smi output:",
        nvml_output,
        "",
    ]
    (REPO / "results" / "audit" / "gpu_diagnostic.txt").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "torch_version", "torch_cuda_runtime", "torch_cuda_available", "torch_cuda_device_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
