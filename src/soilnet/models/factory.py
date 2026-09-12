from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch

from soilnet.final_protocol import normalize_historical_soilnet_state
from soilnet.io import sha256_file
from soilnet.models.baselines import TimmFusionDualHead
from soilnet.models.experiment_soilnet import ExperimentSoilNet
from soilnet.models.soilnet import SoilNetDualHead


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("No safe model state_dict found")


def available_timm_model(model_name: str) -> bool:
    # TIMM accepts explicit pretrained tags (for example `.cvnets_in1k`) even
    # though list_models() enumerates the untagged architecture name.
    return model_name in set(timm.list_models()) or model_name.split(".", 1)[0] in set(timm.list_models())


def _load_verified_soilnet_state(
    model: SoilNetDualHead, checkpoint_path: Path, expected_sha256: str,
    *, source_kind: str, vicreg_checkpoint_loaded: bool,
) -> dict[str, Any]:
    observed = sha256_file(checkpoint_path)
    if observed != expected_sha256:
        raise RuntimeError(f"{source_kind} checkpoint SHA256 mismatch: {observed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw = {str(key).removeprefix("module."): value for key, value in extract_state_dict(checkpoint).items()}
    canonical = model.state_dict()
    raw_missing = sorted(set(canonical) - set(raw))
    raw_unexpected = sorted(set(raw) - set(canonical))
    shape_mismatches = sorted(
        key for key in canonical if key in raw and tuple(canonical[key].shape) != tuple(raw[key].shape)
    )
    normalized, adapter = normalize_historical_soilnet_state(raw, canonical)
    result = model.load_state_dict(normalized, strict=True)
    return {
        "source_kind": source_kind,
        "source_path": str(checkpoint_path),
        "checkpoint_sha256": observed,
        "vicreg_checkpoint_loaded": vicreg_checkpoint_loaded,
        "matched_keys": len(normalized),
        "raw_missing_keys": raw_missing,
        "raw_unexpected_alias_keys": raw_unexpected,
        "shape_mismatches": shape_mismatches,
        "missing_after_normalization": list(result.missing_keys),
        "unexpected_after_normalization": list(result.unexpected_keys),
        **adapter,
    }


def load_verified_soilnet_ssl(model: SoilNetDualHead, checkpoint_path: Path, expected_sha256: str) -> dict[str, Any]:
    return _load_verified_soilnet_state(
        model, checkpoint_path, expected_sha256,
        source_kind="vicreg_ssl", vicreg_checkpoint_loaded=True,
    )


def load_verified_soilnet_initialization(
    model: SoilNetDualHead, checkpoint_path: Path, expected_sha256: str,
) -> dict[str, Any]:
    return _load_verified_soilnet_state(
        model, checkpoint_path, expected_sha256,
        source_kind="historical_pre_vicreg_imagenet_snapshot",
        vicreg_checkpoint_loaded=False,
    )


def build_model(
    config: dict[str, Any], checkpoint_root: Path, *, data_root: Path | None = None,
    run_artifact_root: Path | None = None, pretrained_override: bool | None = None,
):
    architecture = config["architecture"]
    initialization = config["initialization"]
    pretrained = initialization in {"imagenet", "imagenet_then_vicreg"}
    if pretrained_override is not None:
        pretrained = pretrained_override
    load_report = None
    if architecture == "SoilNet":
        use_li = bool(config["use_li"])
        initialization_checkpoint = config.get("initialization_checkpoint")
        if initialization_checkpoint and config.get("ssl_checkpoint"):
            raise RuntimeError("A run cannot load both pre-VICReg initialization and VICReg SSL checkpoints")
        # SSL checkpoints supply all canonical weights; avoid a redundant
        # network download before strict loading. The historical pre-VICReg
        # snapshot likewise supplies the exact ImageNet plus new-layer state.
        model = ExperimentSoilNet(
            num_classes=int(config["num_classes"]),
            use_li_signal=use_li,
            backbone_pretrained=(
                pretrained and initialization == "imagenet" and not initialization_checkpoint
            ),
        )
        ssl = config.get("ssl_checkpoint")
        if ssl:
            load_report = load_verified_soilnet_ssl(model, checkpoint_root / ssl["relative_path"], ssl["sha256"])
        elif initialization_checkpoint:
            if data_root is None:
                raise RuntimeError("DATA_ROOT is required for the historical pre-VICReg initialization")
            load_report = load_verified_soilnet_initialization(
                model,
                data_root / initialization_checkpoint["relative_path"],
                initialization_checkpoint["sha256"],
            )
    else:
        model_name = config.get("timm_model_name")
        if not model_name:
            raise RuntimeError(f"Unresolved timm model for {config['experiment_id']}")
        if not available_timm_model(model_name):
            raise RuntimeError(f"timm model is unavailable in this environment: {model_name}")
        ssl = config.get("ssl_checkpoint")
        model = TimmFusionDualHead(
            model_name,
            num_classes=int(config["num_classes"]),
            # A verified P3 checkpoint supplies the complete backbone state.
            # Avoid a redundant ImageNet download before strict replacement.
            pretrained=pretrained and not ssl,
            use_light=bool(config["use_li"]),
        )
        if ssl:
            if config.get("experiment_id") != "P3_MOBILEVITV2_VICREG_LI_BESTREG":
                raise RuntimeError("Unsupported SSL checkpoint for timm baseline")
            if ssl.get("scope") != "RUN_ARTIFACT_ROOT" or run_artifact_root is None:
                raise RuntimeError("P3 VICReg initialization requires RUN_ARTIFACT_ROOT")
            checkpoint_path = run_artifact_root / ssl["relative_path"]
            observed = sha256_file(checkpoint_path)
            if observed != ssl.get("sha256"):
                raise RuntimeError(f"P3 VICReg checkpoint SHA256 mismatch: {observed}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("backbone_state_dict") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict):
                raise RuntimeError("P3 VICReg checkpoint has no backbone_state_dict")
            expected = model.backbone.state_dict()
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            shape_mismatches = sorted(
                key for key in expected
                if key in state and tuple(expected[key].shape) != tuple(state[key].shape)
            )
            if missing or unexpected or shape_mismatches:
                raise RuntimeError(
                    "P3 VICReg backbone is not strict-compatible: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"shape_mismatches={len(shape_mismatches)}"
                )
            result = model.backbone.load_state_dict(state, strict=True)
            load_report = {
                "source_kind": "mobilevitv2_vicreg_ssl",
                "source_path": str(checkpoint_path),
                "checkpoint_sha256": observed,
                "vicreg_checkpoint_loaded": True,
                "matched_keys": len(state),
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "shape_mismatches": shape_mismatches,
                "encoder_architecture": checkpoint.get("encoder_architecture"),
                "pretraining_epoch": checkpoint.get("epoch"),
                "unlabeled_manifest_sha256": checkpoint.get("unlabeled_manifest_sha256"),
            }
    return model, load_report


def build_frozen_model(config: dict[str, Any]) -> torch.nn.Module:
    """Build architecture only for an already-frozen supervised checkpoint.

    This never downloads or loads initialization weights. The final supervised
    state dict is the only weight source during test and edge deployment.
    """
    architecture = config["architecture"]
    if architecture == "SoilNet":
        return ExperimentSoilNet(
            num_classes=int(config["num_classes"]),
            use_li_signal=bool(config["use_li"]),
            backbone_pretrained=False,
        )
    model_name = config.get("timm_model_name")
    if not model_name or not available_timm_model(model_name):
        raise RuntimeError(f"Frozen architecture is unavailable: {model_name!r}")
    return TimmFusionDualHead(
        model_name,
        num_classes=int(config["num_classes"]),
        pretrained=False,
        use_light=bool(config["use_li"]),
    )


def model_complexity(model: torch.nn.Module, checkpoint_path: Path | None = None) -> dict[str, Any]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.is_file() else 0,
        "input_shape": "3x224x224",
        "li_input_dimension": 1,
        "flops_macs": "NOT_REPORTED_NO_PINNED_RELIABLE_COUNTER",
        "flops_tool": None,
    }
