from __future__ import annotations

import math
from typing import Mapping, Sequence

import domain
from aero_log_parser import TrajectoryPoint

# -----------------------
# Soft-constraint helpers
# -----------------------

TINY = 1e-12


def _pos(x: float) -> float:
    return x if x > 0.0 else 0.0


def v_le(value: float, upper: float, scale: float) -> float:
    """Violation of value <= upper (soft)."""
    return _pos((value - upper) / scale)


def v_ge(value: float, lower: float, scale: float) -> float:
    """Violation of value >= lower (soft)."""
    return _pos((lower - value) / scale)


def v_box(value: float, lo: float, hi: float, scale: float) -> float:
    """Violation of lo <= value <= hi (soft)."""
    return v_ge(value, lo, scale) + v_le(value, hi, scale)


def _cget(const: Mapping[str, float], key: str, default: float) -> float:
    return float(const.get(key, default))


def _penalty(const: Mapping[str, float], key: str, default: float = 1e3) -> float:
    return float(const.get(key, default))


# -----------------------
# Constraint groups
# -----------------------

def violations_type1(ad: domain.AerodynamicsInfo, const: Mapping[str, float]) -> dict[str, float]:
    sL = _cget(const, "sL", 0.1)

    L_warhead_start = const["L_warhead_start"]
    L_head = const["L_head"]
    L_0 = const["L_0"]
    L_stern = const["L_stern"]

    c1 = v_le(ad.L_st_position + ad.L_st, L_warhead_start, sL)
    c2 = v_ge(ad.L_st_position, L_head, sL)

    x = ad.L_w_position + ad.L_w
    lo = L_0 - L_stern
    hi = L_0 - L_stern / 2.0
    c3 = v_box(x, lo, hi, sL)

    return {"c1": c1, "c2": c2, "c3": c3}


def violations_type2(wr: domain.WeightsCalculationResult, const: Mapping[str, float]) -> dict[str, float]:
    sM = _cget(const, "sM", 1.0)
    m0 = const["m_0inp"]
    return {"c4": v_box(wr.m_0, m0 - 10.0, m0 + 10.0, sM)}


def violations_type3(
        br: domain.BallisticsCalculationResult,
        wr: domain.WeightsCalculationResult,
        const: Mapping[str, float],
) -> dict[str, float]:
    sM = _cget(const, "sM", 1.0)
    sV = _cget(const, "sV", 10.0)
    sTheta = _cget(const, "sTheta", math.radians(1.0))
    sMdot = _cget(const, "sMdot", 0.1)

    V_target = const["V_target"]
    V_lo = 1.5 * V_target
    V_hi = 2.5 * V_target

    eps_theta = _cget(const, "eps_theta", math.radians(0.5))

    return {
        "c5": v_ge(br.mdot_final, 0.0, sMdot),  # mdot_final >= 0
        "c6": v_le(br.theta_final, math.pi / 2.0 - eps_theta, sTheta),  # theta_final < 90deg
        "c7": v_ge(br.m_final, wr.m_k, sM),  # m_final >= m_k
        "c8": v_box(br.v_final, V_lo, V_hi, sV),  # velocity band
    }


def violations_type4_points(
        points: Sequence[TrajectoryPoint],
        ad: domain.AerodynamicsInfo,
        const: Mapping[str, float],
) -> dict[str, float]:
    """Max-over-points violations for c9..c12 for a single trajectory."""
    if not points:
        big = _penalty(const, "penalty_no_points")
        return {"c9": big, "c10": big, "c11": big, "c12": big}

    g = 9.80665
    alpha_max = math.radians(_cget(const, "alpha_max_deg", 13.0))
    n_ymax = const["n_ymax"]
    jn_max = const["j_n_max"]

    S_m = math.pi * (ad.d_M ** 2) / 4.0  # reference area

    sAlpha = _cget(const, "sAlpha", math.radians(1.0))
    sNy = _cget(const, "sNy", 1.0)
    sCya = _cget(const, "sCya", 0.1)
    sR = _cget(const, "sR", 0.1)

    pen_bad_denom = _penalty(const, "penalty_bad_denom")
    pen_div0_mz = _penalty(const, "penalty_div0_mz")

    max_c9 = 0.0
    max_c10 = 0.0
    max_c11 = 0.0
    sum_c12 = 0.0

    for pt in points:
        # (9) alpha <= alpha_max
        c9 = v_le(pt.alpha, alpha_max, sAlpha)

        # (10) n_y = Jn/g <= n_ymax
        n_y = pt.Jn / g
        c10 = v_le(n_y, n_ymax, sNy)

        # (11) Cya_needed <= Cy_alpha_traj
        denom = pt.q * S_m * alpha_max
        if pt.q <= 0.0 or S_m <= 0.0 or abs(denom) < TINY:
            c11 = pen_bad_denom
        else:
            cya_needed = (jn_max * pt.m + pt.m * g * math.cos(pt.theta) - pt.P * math.sin(pt.alpha)) / denom
            c11 = v_le(cya_needed, pt.Cy_alpha_traj, sCya)

        # (12) 1.5 <= |mz_alpha|/|mz_delta| <= 2.5
        mz_d = abs(pt.mz_delta)
        if mz_d < TINY:
            c12 = pen_div0_mz
        else:
            r = abs(pt.mz_alpha) / mz_d
            c12 = v_box(r, 1.5, 2.5, sR)

        max_c9 = max(max_c9, c9)
        max_c10 = max(max_c10, c10)
        max_c11 = max(max_c11, c11)
        sum_c12 += c12

    return {
        "c9": max_c9,
        "c10": max_c10,
        "c11": max_c11,
        "c12": sum_c12 / len(points),
    }
