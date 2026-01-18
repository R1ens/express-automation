import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cma  # pip install cma

import domain
import calc


# ============================================================
# INPUTS (set these once)
# ============================================================

m_0 = 700.0
m_fuel = 315.0
m_empty = m_0 - m_fuel

d_middle = 0.4
V_0 = 375.0
J = 2100.0
d_nozzle = 0.25
t_engine = 43.864219

V_target = 340.0
V_LO = 1.5 * V_target
V_HI = 2.5 * V_target

mdot_A = 13.404203
mdot_B = -0.294651

PHI_BASE_DEG = 12.5

PHI_MIN_DEG = 12.0
PHI_MAX_DEG = 89.8

X_TOL = 1e-3


POINTS = [
    {"id": 1, "Y": 9800.0, "X_set": 46105.37507, "340ode": "fixed", "phi_deg": PHI_BASE_DEG},
    {"id": 2, "Y": 900.0,  "X_set": 830, "mode": "solve"},
    {"id": 3, "Y": 9800.0, "X_set": 830,  "mode": "solve"},
    {"id": 4, "Y": 900.0,  "X_set": 4234.167099, "8713,55": "fixed", "phi_deg": PHI_BASE_DEG},
]


# ============================================================
# CHECKS
# ============================================================

@dataclass(frozen=True)
class CheckFlags:
    mdot_ok: bool
    theta_ok: bool
    mass_ok: bool
    velocity_ok: bool

    @property
    def ok(self) -> bool:
        return self.mdot_ok and self.theta_ok and self.mass_ok and self.velocity_ok


def check_detailed(r: domain.CalculationResult) -> CheckFlags:
    return CheckFlags(
        mdot_ok=(r.mdot_final >= 0),
        theta_ok=(r.theta_final >= 0),
        mass_ok=(r.m_final >= m_empty),
        velocity_ok=(V_LO <= r.v_final <= V_HI),
    )


# ============================================================
# CSV LOGGING
# ============================================================

ALL_CSV = Path("theta_points_all.csv")
BEST_CSV = Path("theta_points_best.csv")

FIELDS = [
    "ts", "point_id", "mode",
    "phi_deg", "phi_rad",
    "Y", "X_set", "X_geom",
    "X_final", "X_err", "abs_X_err",
    "ok", "mdot_ok", "theta_ok", "mass_ok", "velocity_ok",
    "m_final", "mdot_final", "v_final", "theta_final",
    "trajectory_log_forces", "trajectory_log_position",
    "error",
]

def ensure_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def append_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

ensure_header(ALL_CSV)
ensure_header(BEST_CSV)


# ============================================================
# CORE EVALUATION
# ============================================================

def x_geom_from_phi(Y: float, phi_rad: float) -> float:
    t = math.tan(phi_rad)
    if not math.isfinite(t) or abs(t) < 1e-12:
        return float("nan")
    return float(Y) / t

def build_la(phi_rad: float) -> domain.LAInfo:
    return domain.LAInfo(
        m_0=float(m_0),
        d_middle=float(d_middle),
        mdot_A=float(mdot_A),
        mdot_B=float(mdot_B),
        t_engine=float(t_engine),
        d_nozzle=float(d_nozzle),
        J=float(J),
        V_0=float(V_0),
        theta_0=float(phi_rad),
        aerodynamics='2.ad',
    )

def eval_point(point_id: int, mode: str, Y: float, X_set: float, phi_rad: float) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (abs_error, payload)
    - abs_error is finite always (penalized for infeasible/errors)
    - payload has ok flag; "best" logic will only accept ok==1
    """
    X_geom = x_geom_from_phi(Y, phi_rad)

    payload: Dict[str, Any] = {
        "ts": time.time(),
        "point_id": point_id,
        "mode": mode,
        "phi_deg": float(np.rad2deg(phi_rad)),
        "phi_rad": float(phi_rad),
        "Y": float(Y),
        "X_set": float(X_set),
        "X_geom": float(X_geom),
        "X_final": float("nan"),
        "X_err": float("nan"),
        "abs_X_err": float("nan"),
        "ok": 0,
        "mdot_ok": 0,
        "theta_ok": 0,
        "mass_ok": 0,
        "velocity_ok": 0,
        "m_final": float("nan"),
        "mdot_final": float("nan"),
        "v_final": float("nan"),
        "theta_final": float("nan"),
        "trajectory_log_forces": "",
        "trajectory_log_position": "",
        "error": "",
    }

    # Hard penalty for invalid geometry
    if not np.isfinite(X_geom):
        payload["error"] = "X_geom_invalid"
        append_row(ALL_CSV, payload)
        return 1e12, payload

    la = build_la(phi_rad)
    tgt = domain.TargetInfo(velocity=V_target, x=float(X_geom), y=float(Y))

    try:
        r = calc.calculate_ballistics(tgt, la, domain.IntegrationInfo())
        flags = check_detailed(r)

        payload.update({
            "X_final": float(r.X_final),
            "X_err": float(r.X_final - X_set),
            "abs_X_err": float(abs(r.X_final - X_set)),
            "ok": int(flags.ok),
            "mdot_ok": int(flags.mdot_ok),
            "theta_ok": int(flags.theta_ok),
            "mass_ok": int(flags.mass_ok),
            "velocity_ok": int(flags.velocity_ok),
            "m_final": float(r.m_final),
            "mdot_final": float(r.mdot_final),
            "v_final": float(r.v_final),
            "theta_final": float(r.theta_final),
            "trajectory_log_forces": getattr(r, "trajectory_log_forces", "") or "",
            "trajectory_log_position": getattr(r, "trajectory_log_position", "") or "",
        })

        append_row(ALL_CSV, payload)

        # Only feasible points are considered valid; infeasible gets a big penalty
        if not flags.ok:
            payload["error"] = "failed_checks"
            # penalty grows a bit with how far from target you are, but dominated by infeasibility
            base = float(payload["abs_X_err"]) if np.isfinite(payload["abs_X_err"]) else 1e6
            return 1e10 + base, payload

        return float(payload["abs_X_err"]), payload

    except Exception as e:
        payload["error"] = f"exception:{type(e).__name__}:{e}"
        append_row(ALL_CSV, payload)
        return 1e12, payload


import math
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# Uses your existing:
# - eval_point(point_id, mode, Y, X_set, phi_rad) -> (J, payload)   (J finite or penalized ok)
# - append_row(BEST_CSV, payload) in record_best if you want
# - PHI_MIN_DEG / PHI_MAX_DEG / X_TOL

def _eval_signed(point_id: int, mode: str, Y: float, X_set: float, phi: float) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Returns (f, payload)
    f = X_final - X_set for feasible only, else None.
    """
    J, payload = eval_point(point_id, mode, Y, X_set, phi)
    if payload.get("ok", 0) == 1 and np.isfinite(payload.get("X_err", np.nan)):
        return float(payload["X_err"]), payload
    return None, payload


def solve_phi_bisection(
    point_id: int,
    Y: float,
    X_set: float,
    *,
    max_probe: int = 40,      # tries to find a feasible sign-change bracket
    max_iter: int = 60,       # bisection iterations once bracketed
    mid_rescue_tries: int = 10,  # if midpoint infeasible, try nearby points
    seed: Optional[int] = 0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Goal: nearest-to-X_set feasible solution.
    Strategy:
      1) Randomly probe inside bounds to find:
         - best feasible so far (min abs err)
         - a feasible sign-change bracket (a,b) with f(a)*f(b) < 0
      2) If bracket found -> bisection, always tracking best feasible
      3) If no bracket -> return best feasible found from probes
    """

    rng = np.random.default_rng(seed)

    phi_lo = float(np.deg2rad(PHI_MIN_DEG))
    phi_hi = float(np.deg2rad(PHI_MAX_DEG))

    best_abs = float("inf")
    best_payload: Optional[Dict[str, Any]] = None

    def record_best(payload: Dict[str, Any]) -> None:
        nonlocal best_abs, best_payload
        if payload.get("ok", 0) != 1:
            return
        ae = float(payload.get("abs_X_err", float("inf")))
        if ae < best_abs - 1e-15:
            best_abs = ae
            best_payload = payload
            append_row(BEST_CSV, payload)  # keep if you want "best updates"

    # ---- Phase 1: probe to find feasible points + sign-change bracket ----
    feasible_samples: List[Tuple[float, float, Dict[str, Any]]] = []  # (phi, f, payload)

    # include endpoints as probes too (sometimes feasible)
    for phi in (phi_lo, phi_hi):
        f, payload = _eval_signed(point_id, "bisect_probe", Y, X_set, phi)
        record_best(payload)
        if f is not None:
            feasible_samples.append((phi, f, payload))

    # random probes (NOT a grid)
    for _ in range(max_probe):
        phi = float(rng.uniform(phi_lo, phi_hi))
        f, payload = _eval_signed(point_id, "bisect_probe", Y, X_set, phi)
        record_best(payload)
        if f is not None:
            feasible_samples.append((phi, f, payload))

        # try to form a bracket opportunistically
        if len(feasible_samples) >= 2:
            # check against last few feasible points
            phi_new, f_new, _ = feasible_samples[-1]
            for phi_old, f_old, _ in feasible_samples[-10:]:
                if f_old == 0.0:
                    return phi_old, best_payload if best_payload is not None else payload
                if f_old * f_new < 0.0:
                    a, fa = (phi_old, f_old)
                    b, fb = (phi_new, f_new)
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    bracket = (a, fa, b, fb)
                    break
            else:
                bracket = None
        else:
            bracket = None

        if bracket is not None:
            break
    else:
        bracket = None

    if best_payload is None:
        raise RuntimeError(
            f"Point {point_id}: no feasible points found in {max_probe} probes. "
            f"Increase max_probe or widen PHI bounds, or feasibility may be impossible."
        )

    # If no sign-change bracket, we can’t do bisection properly.
    # Return best feasible (nearest) found.
    if bracket is None:
        return float(best_payload["phi_rad"]), best_payload

    a, fa, b, fb = bracket

    # ---- Phase 2: bisection inside feasible sign-change bracket ----
    for _ in range(max_iter):
        # stop if we already have a good enough feasible solution
        if float(best_payload.get("abs_X_err", 1e99)) <= float(X_TOL):
            break

        mid = 0.5 * (a + b)

        fmid, pmid = _eval_signed(point_id, "bisect_mid", Y, X_set, mid)
        record_best(pmid)

        if fmid is None:
            # Midpoint infeasible: try "rescue" points near the midpoint
            width = (b - a)
            rescued = False
            for k in range(1, mid_rescue_tries + 1):
                # symmetric offsets shrinking with k
                frac = 0.5 ** k
                for sgn in (-1.0, +1.0):
                    phi_try = mid + sgn * frac * 0.5 * width
                    if not (a < phi_try < b):
                        continue
                    f_try, p_try = _eval_signed(point_id, "bisect_rescue", Y, X_set, phi_try)
                    record_best(p_try)
                    if f_try is not None:
                        mid = phi_try
                        fmid = f_try
                        rescued = True
                        break
                if rescued:
                    break

            if not rescued:
                # Could not get a feasible midpoint; shrink bracket toward the side
                # where we *already know* it is feasible by pulling in 25% from both ends.
                a = a + 0.25 * (b - a)
                b = b - 0.25 * (b - a)
                # Refresh ends (keep feasible ends if possible)
                fa2, pa = _eval_signed(point_id, "bisect_end", Y, X_set, a)
                fb2, pb = _eval_signed(point_id, "bisect_end", Y, X_set, b)
                record_best(pa); record_best(pb)
                if fa2 is not None: fa = fa2
                if fb2 is not None: fb = fb2
                continue

        # fmid is feasible here
        if fmid == 0.0:
            break

        # standard bisection update
        if fa * fmid < 0.0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

    return float(best_payload["phi_rad"]), best_payload


# ============================================================
# FINAL OUTPUT TABLE
# ============================================================

def print_final_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "pt", "phi(deg)", "Y", "X_set", "X_geom", "X_final", "X_err",
        "ok", "mdot", "theta", "mass", "vel",
        "m_final", "mdot_f", "v_final", "theta_f",
    ]
    print("\n" + " | ".join(headers))
    print("-" * 160)

    for r in rows:
        print(
            f"{int(r['point_id']):>2d} | "
            f"{r['phi_deg']:>7.3f} | "
            f"{r['Y']:>7.1f} | "
            f"{r['X_set']:>10.3f} | "
            f"{r['X_geom']:>10.3f} | "
            f"{r['X_final']:>10.3f} | "
            f"{r['X_err']:>+9.3f} | "
            f"{int(r['ok']):>2d} | "
            f"{int(r['mdot_ok']):>4d} | "
            f"{int(r['theta_ok']):>5d} | "
            f"{int(r['mass_ok']):>4d} | "
            f"{int(r['velocity_ok']):>3d} | "
            f"{r['m_final']:>8.3f} | "
            f"{r['mdot_final']:>7.3f} | "
            f"{r['v_final']:>7.3f} | "
            f"{r['theta_final']:>7.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Running 4-point theta script (CMA-ES for solve)...")
    print(f"Logs: {ALL_CSV} (all), {BEST_CSV} (best feasible updates)")

    final_rows: List[Dict[str, Any]] = []

    for p in POINTS:
        pid = int(p["id"])
        Y = float(p["Y"])
        X_set = float(p["X_set"])
        mode = str(p["mode"])

        if mode == "fixed":
            phi_deg = float(p.get("phi_deg", PHI_BASE_DEG))
            phi_rad = float(np.deg2rad(phi_deg))
            _, payload = eval_point(pid, "fixed", Y, X_set, phi_rad)

            if payload.get("ok", 0) != 1:
                print(f"⚠️ Point {pid} fixed(phi={phi_deg}) failed checks or errored: {payload.get('error','')}")
            append_row(BEST_CSV, payload)
            final_rows.append(payload)

        elif mode == "solve":
            print(f"Solving point {pid} (Y={Y}, X_set={X_set}) with bisection ...")
            phi_rad, payload = solve_phi_bisection(
                pid, Y, X_set,
                max_probe=60,  # increase if feasibility is rare
                max_iter=100,
                mid_rescue_tries=15,
                seed=pid
            )
            final_rows.append(payload)

        else:
            raise ValueError(f"Unknown mode for point {pid}: {mode}")

    print_final_table(final_rows)

    print("\nDone.")
    print(f"CSV written:\n - {ALL_CSV.resolve()}\n - {BEST_CSV.resolve()}")


if __name__ == "__main__":
    main()
