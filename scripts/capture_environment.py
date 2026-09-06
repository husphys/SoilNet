#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from soilnet.io import resolve_paths, write_json


def command(argv):
    try:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return result.returncode, result.stdout.strip()
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def main() -> int:
    configured = resolve_paths()
    commands = [
        ["uname", "-a"], ["wsl.exe", "--version"], ["nvidia-smi"],
        ["python3", "--version"], [sys.executable, "--version"], ["git", "--version"],
        ["git", "lfs", "version"], ["conda", "--version"], ["mamba", "--version"],
        ["df", "-h", str(REPO), str(configured["checkpoint_root"]), str(configured["data_root"])],
    ]
    lines = [f"captured_utc: {datetime.now(timezone.utc).isoformat()}"]
    for argv in commands:
        code, output = command(argv)
        lines.extend([f"$ {' '.join(argv)}", f"exit_code: {code}", output, ""])
    (REPO / "results" / "audit" / "system_info.txt").write_text("\n".join(lines), encoding="utf-8")

    environment = {
        "classification": "tested reproduction environment, not proven original environment",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }
    for module_name in ("torch", "torchvision", "torchaudio", "timm", "numpy", "pandas", "sklearn", "PIL", "yaml"):
        try:
            module = __import__(module_name)
            environment[module_name] = getattr(module, "__version__", "installed-version-not-exposed")
        except Exception as exc:
            environment[module_name] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    try:
        import torch
        environment.update({
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE",
        })
    except Exception:
        pass
    write_json(REPO / "results" / "audit" / "environment.json", environment)

    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (REPO / "requirements-lock.txt").write_text(freeze, encoding="utf-8")
    conda = shutil.which("conda")
    if conda:
        exported = subprocess.check_output([conda, "env", "export", "--prefix", sys.prefix, "--from-history"], text=True)
        exported_lines = [line for line in exported.splitlines() if not line.startswith("prefix:")]
        if not any(line.startswith("name:") for line in exported_lines):
            exported_lines.insert(0, "name: soilnet")
        exported = "\n".join(exported_lines) + "\n"
        (REPO / "environment.yml").write_text(exported, encoding="utf-8")
    print(json.dumps(environment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
