from __future__ import annotations

from typing import Dict, Tuple


def sumsq(d: Dict[str, float], w: Dict[str, float] | None = None) -> float:
    total = 0.0
    for k, v in d.items():
        wk = 1.0 if w is None else float(w.get(k, 1.0))
        total += wk * (float(v) ** 2)
    return total


def max_merge(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    return {k: max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys}


def avg_merge(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    return {
        k: 0.5 * (float(a.get(k, 0.0)) + float(b.get(k, 0.0)))
        for k in keys
    }


def aggregate_score(
        v_type1: Dict[str, float],
        v_type2: Dict[str, float],
        v_type3_A: Dict[str, float],
        v_type3_B: Dict[str, float],
        v_type4_A: Dict[str, float],
        v_type4_B: Dict[str, float],
        *,
        weights: Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Trajectory-combine policy:
      - type3_combined = max per constraint between A and B
      - type4_combined = average per constraint between A and B
    """
    type3 = max_merge(v_type3_A, v_type3_B)
    type4 = avg_merge(v_type4_A, v_type4_B)

    parts = {
        "type1": sumsq(v_type1, weights),
        "type2": sumsq(v_type2, weights),
        "type3": sumsq(type3, weights),
        "type4": sumsq(type4, weights),
    }
    total = sum(parts.values())

    detail = {
        "type1": v_type1,
        "type2": v_type2,
        "type3_A": v_type3_A,
        "type3_B": v_type3_B,
        "type3_combined": type3,
        "type4_A": v_type4_A,
        "type4_B": v_type4_B,
        "type4_combined": type4,
    }

    return total, parts, detail
