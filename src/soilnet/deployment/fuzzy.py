from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if a <= x <= b:
        return 1.0 if b == a else (x - a) / (b - a)
    if b < x <= c:
        return 1.0
    if c < x <= d:
        return 1.0 if d == c else (d - x) / (d - c)
    return 0.0


def _triangle(x: float, a: float, b: float, c: float) -> float:
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if a <= x < b:
        return 1.0 if b == a else (x - a) / (b - a)
    if b < x <= c:
        return 1.0 if c == b else (c - x) / (c - b)
    return 0.0


@dataclass(frozen=True)
class FuzzyDecision:
    duration_seconds: int
    duration_minutes_raw: float
    memberships_sm0: dict[str, float]
    memberships_sm20: dict[str, float]
    firing_strengths: dict[str, float]


class SugenoIrrigationController:
    labels = ("dry", "moderate", "wet")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rule_minutes = config["rules_minutes"]
        self.max_seconds = int(config["maximum_duration_seconds"])

    def _memberships(self, x: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for label in self.labels:
            spec = self.config[label]
            points = [float(v) for v in spec["points"]]
            if spec["type"] == "trapezoid":
                value = _trapezoid(x, *points)
            elif spec["type"] == "triangle":
                value = _triangle(x, *points)
            else:
                raise ValueError(f"Unsupported membership type: {spec['type']}")
            out[label] = min(1.0, max(0.0, float(value)))
        return out

    def evaluate(self, sm0: float, sm20: float) -> FuzzyDecision:
        if not (math.isfinite(sm0) and math.isfinite(sm20)):
            return FuzzyDecision(0, 0.0, {}, {}, {})

        x0 = min(100.0, max(0.0, float(sm0)))
        x20 = min(100.0, max(0.0, float(sm20)))
        m0 = self._memberships(x0)
        m20 = self._memberships(x20)

        weighted_sum = 0.0
        weight_total = 0.0
        strengths: dict[str, float] = {}

        for a in self.labels:
            for b in self.labels:
                weight = m0[a] * m20[b]
                strengths[f"{a}_{b}"] = weight
                weighted_sum += weight * float(self.rule_minutes[a][b])
                weight_total += weight

        minutes = 0.0 if weight_total <= 0.0 else weighted_sum / weight_total
        seconds = min(self.max_seconds, max(0, int(round(minutes * 60.0))))

        return FuzzyDecision(
            duration_seconds=seconds,
            duration_minutes_raw=minutes,
            memberships_sm0=m0,
            memberships_sm20=m20,
            firing_strengths=strengths,
        )
