from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import cma
import numpy as np
import domain
from aero_eval import evaluate_candidate

# -----------------------
# Variable mapping
# -----------------------

VAR_ORDER: List[str] = [
    "L_st_position",
    "L_st_span",
    "L_st",
    "L_st_straight",
    "delta_st",
    "L_w_position",
    "L_w_span",
    "L_w",
    "L_w_straight",
    "delta_w",
]


def ensure_out_dir() -> Path:
    out_dir = Path("out") / "aerodynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_best_base(csv_best: Path) -> Optional[Dict[str, float]]:
    """
    Read cma_best.csv and return BASE values from the row with minimal 'score'
    (global best across all appended runs).

    Returns None if file missing/empty/unparseable or required fields missing.
    """
    if not csv_best.exists():
        return None

    try:
        best_row: Optional[Dict[str, str]] = None
        best_score = float("inf")

        with csv_best.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                s_raw = row.get("score", "")
                if s_raw is None or str(s_raw).strip() == "":
                    continue
                try:
                    s = float(s_raw)
                except ValueError:
                    continue

                if s < best_score:
                    # ensure the row has all vars (otherwise skip it)
                    ok = True
                    for k in VAR_ORDER:
                        v = row.get(k, None)
                        if v is None or str(v).strip() == "":
                            ok = False
                            break
                    if not ok:
                        continue
                    best_score = s
                    best_row = row

        if not best_row:
            return None

        base_override: Dict[str, float] = {k: float(best_row[k]) for k in VAR_ORDER}
        return base_override

    except Exception:
        return None


def make_bounds(base_override: Optional[Dict[str, float]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns LOW, BASE, HIGH arrays in VAR_ORDER. BASE can be overridden from last cma_best.csv row."""
    bounds = {
        "L_st_position": (0.7875, 0.9375, 1.0875),
        "L_st_span": (0.2, 0.60, 1.5),
        "L_st": (0.02, 0.20, 0.35),
        "L_st_straight": (0.01, 0.08, 0.4),
        "delta_st": (0.005, 0.02, 0.03),
        "L_w_position": (2, 3.8, 7),
        "L_w_span": (0.4, 1.7, 2.5),
        "L_w": (0.3, 1.6, 2.5),
        "L_w_straight": (0.2, 0.50, 0.9),
        "delta_w": (0.01, 0.03, 0.04),
    }

    # override BASE (middle value) from last best row, while keeping LOW/HIGH fixed
    if base_override:
        for k, v in base_override.items():
            if k in bounds:
                lo, _, hi = bounds[k]
                # keep it sane if previous run wrote something slightly outside
                vv = float(np.clip(float(v), lo, hi))
                bounds[k] = (lo, vv, hi)

    low = np.array([bounds[k][0] for k in VAR_ORDER], dtype=float)
    base = np.array([bounds[k][1] for k in VAR_ORDER], dtype=float)
    high = np.array([bounds[k][2] for k in VAR_ORDER], dtype=float)
    return low, base, high


def u_to_x(u: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return low + u * (high - low)


def x_to_named(x: np.ndarray) -> Dict[str, float]:
    return {k: float(x[i]) for i, k in enumerate(VAR_ORDER)}


# -----------------------
# Logging / flattening
# -----------------------

def flatten_row(
        *,
        ts: float,
        eval_id: int,
        gen: int,
        idx: int,
        u: np.ndarray,
        x10_named: Dict[str, float],
        score: float,
        diag: Dict[str, Any],
) -> Dict[str, Any]:
    score_parts = diag.get("score_parts") or {}
    viol = diag.get("violations") or {}
    type4 = viol.get("type4_combined") or {}

    t1 = diag.get("traj1") or {}
    t2 = diag.get("traj2") or {}

    row: Dict[str, Any] = {
        # meta
        "ts": ts,
        "eval_id": eval_id,
        "gen": gen,
        "idx": idx,
        "score": float(score),

        # score parts
        "type1": float(score_parts.get("type1", 0.0)),
        "type2": float(score_parts.get("type2", 0.0)),
        "type3": float(score_parts.get("type3", 0.0)),
        "type4": float(score_parts.get("type4", 0.0)),

        # type4 combined
        "c9": float(type4.get("c9", 0.0)),
        "c10": float(type4.get("c10", 0.0)),
        "c11": float(type4.get("c11", 0.0)),
        "c12": float(type4.get("c12", 0.0)),
    }

    # inputs (physical)
    for k in VAR_ORDER:
        row[k] = float(x10_named[k])

    # raw u
    for i, k in enumerate(VAR_ORDER):
        row[f"u_{k}"] = float(u[i])

    # traj summaries
    def put_traj(prefix: str, tj: Dict[str, Any]) -> None:
        row[f"{prefix}_n_points"] = int(tj.get("n_points", 0) or 0)
        row[f"{prefix}_X_final"] = float(tj.get("X_final", float("nan")))
        row[f"{prefix}_m_final"] = float(tj.get("m_final", float("nan")))
        row[f"{prefix}_mdot_final"] = float(tj.get("mdot_final", float("nan")))
        row[f"{prefix}_v_final"] = float(tj.get("v_final", float("nan")))
        row[f"{prefix}_theta_final"] = float(tj.get("theta_final", float("nan")))

    put_traj("traj1", t1)
    put_traj("traj2", t2)

    return row


def build_fieldnames() -> List[str]:
    # fixed, stable order = safer CSVs
    fields: List[str] = [
        "ts", "eval_id", "gen", "idx", "score",
        "type1", "type2", "type3", "type4",
        "c9", "c10", "c11", "c12",
    ]
    fields += VAR_ORDER
    fields += [f"u_{k}" for k in VAR_ORDER]
    fields += [
        "traj1_n_points", "traj1_X_final", "traj1_m_final", "traj1_mdot_final", "traj1_v_final", "traj1_theta_final",
        "traj2_n_points", "traj2_X_final", "traj2_m_final", "traj2_mdot_final", "traj2_v_final", "traj2_theta_final",
    ]
    return fields


@dataclass
class BestSoFar:
    score: float = float("inf")
    u: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    diag: Optional[Dict[str, Any]] = None
    row: Optional[Dict[str, Any]] = None  # store last flattened row for best


def main() -> None:
    # -----------------------
    # Project constants
    # -----------------------
    const = {
        "m_0inp": 500.0,
        "L_0": 4.1705,
        "L_head": 0.7875,
        "L_stern": 0.6217,
        "L_warhead_start": 1.0875,
        "V_target": 270.0,
        "n_ymax": 12.0,

        "j_n_max": 117.72,
        "alpha_max_deg": 13.0,

        "d_M": 0.35,
        "d_stern": 0.3,
        "L_cm": 0.0,

        # scales
        "sL": 0.1,
        "sM": 1.0,
        "sV": 10.0,
        "sTheta": np.deg2rad(1.0),
        "sMdot": 0.1,
        "sAlpha": np.deg2rad(1.0),
        "sNy": 1.0,
        "sCya": 0.1,
        "sR": 0.1,

        # penalties
        "penalty_ballistics_fail": 1e3,
        "penalty_no_points": 1e3,
        "penalty_bad_denom": 1e3,
        "penalty_div0_mz": 1e3,
    }

    la_base = domain.LAInfo(
        m_0=500.0,
        d_middle=0.35,
        mdot_A=8.000000,
        mdot_B=-0.080000,
        t_engine=32.643060,
        d_nozzle=0.25,
        J=1968.0,
        V_0=390.0,
        theta_0=np.deg2rad(12.0),
        aerodynamics="unused_here.ad",
    )

    w_info = domain.WeightInfo(
        m_body=227.5732,
        a_body=413.9625,
        m_fuel=225.0,
        x_cm_fuel=3.0490,
        rho_w=1733,
        rho_st=2600,
    )

    # Targets
    V_target = const["V_target"]
    theta_0 = la_base.theta_0
    Y_min, Y_max = 700.0, 7300.0
    X_1, Y_1 = Y_max / np.tan(theta_0), Y_max
    X_2, Y_2 = Y_min / np.tan(theta_0), Y_min

    targets = [
        domain.TargetInfo(velocity=V_target, x=float(X_1), y=float(Y_1)),
        domain.TargetInfo(velocity=V_target, x=float(X_2), y=float(Y_2)),
    ]

    integration = domain.IntegrationInfo()

    out_dir = ensure_out_dir()
    csv_best = out_dir / "cma_best.csv"
    base_override = read_best_base(csv_best)
    if base_override:
        print(f"Seeding BASE bounds from GLOBAL best row in {csv_best} (min score).")
        print({k: base_override[k] for k in VAR_ORDER})
    else:
        print("No previous best found (or unreadable). Using hardcoded BASE bounds.")

    low, base, high = make_bounds(base_override=base_override)

    # CMA in u-space [0,1]^10
    u0 = np.clip((base - low) / (high - low), 0.0, 1.0)
    sigma0 = 0.25

    best = BestSoFar()

    csv_all = out_dir / "cma_all.csv"

    fieldnames = build_fieldnames()

    # CMA options
    opts = {
        "bounds": [0.0, 1.0],
        "popsize": 18,
        "verb_disp": 1,
        "verb_log": 0,
        "tolfun": 0.0,
        "tolx": 1e-12,
    }
    es = cma.CMAEvolutionStrategy(u0.tolist(), sigma0, opts)

    eval_counter = 0
    max_generations = 200

    def eval_one(u: np.ndarray, gen: int, idx: int) -> tuple[float, Dict[str, Any], Dict[str, Any], np.ndarray]:
        nonlocal eval_counter
        u = np.asarray(u, dtype=float)
        x = u_to_x(u, low, high)
        x10 = x_to_named(x)

        score, diag = evaluate_candidate(
            x10_named=x10,
            const=const,
            weight_info=w_info,
            la_info_base=la_base,
            targets=targets,
            integration=integration,
            aero_dir=".",
        )

        eval_counter += 1
        ts = time.time()
        row = flatten_row(
            ts=ts,
            eval_id=eval_counter,
            gen=gen,
            idx=idx,
            u=u,
            x10_named=x10,
            score=float(score),
            diag=diag,
        )
        return float(score), diag, row, x

    # Keep CSV files open for performance
    write_all_header = not csv_all.exists()
    write_best_header = not csv_best.exists()

    with csv_all.open("a", newline="", encoding="utf-8") as fa, csv_best.open("a", newline="", encoding="utf-8") as fb:
        wa = csv.DictWriter(fa, fieldnames=fieldnames, extrasaction="ignore")
        wb = csv.DictWriter(fb, fieldnames=fieldnames, extrasaction="ignore")
        if write_all_header:
            wa.writeheader()
        if write_best_header:
            wb.writeheader()

        for gen in range(max_generations):
            X = es.ask()
            F: List[float] = []

            for idx, ui in enumerate(X):
                score, diag, row, x = eval_one(np.array(ui, dtype=float), gen, idx)

                wa.writerow(row)
                fa.flush()

                if score < best.score:
                    best.score = score
                    best.u = np.array(ui, dtype=float)
                    best.x = x.copy()
                    best.diag = diag
                    best.row = row

                    wb.writerow(row)
                    fb.flush()

                    print("\nNEW BEST SCORE:", best.score)
                    print("SCORE PARTS:", diag.get("score_parts"))
                    print("TYPE4 COMBINED:", (diag.get("violations") or {}).get("type4_combined"))
                    print("TRAJ1:", diag.get("traj1"))
                    print("TRAJ2:", diag.get("traj2"))
                    print("x10_named:", {k: row[k] for k in VAR_ORDER})

                F.append(score)

            es.tell(X, F)
            es.disp()

            # your explicit stop
            if best.score <= 0.0:
                print("\n✅ FEASIBLE POINT FOUND (score==0). Stopping.")
                break

            # CMA internal stopping criteria (optional, but usually useful)
            if es.stop():
                print("\n🛑 CMA stop criteria triggered:", es.stop())
                break

    if best.x is None:
        print("No evaluations performed.")
        return

    best_named = x_to_named(best.x)
    print("\n=== BEST FOUND ===")
    print("score:", best.score)
    print("x10_named:", best_named)
    if best.diag:
        print("score_parts:", best.diag.get("score_parts"))
        print("type4_combined:", (best.diag.get("violations") or {}).get("type4_combined"))
        print("traj1:", best.diag.get("traj1"))
        print("traj2:", best.diag.get("traj2"))

    out = {
        "score": best.score,
        "x10_named": best_named,
        "score_parts": (best.diag or {}).get("score_parts"),
        "violations": (best.diag or {}).get("violations", {}),
        "traj1": (best.diag or {}).get("traj1"),
        "traj2": (best.diag or {}).get("traj2"),
    }
    with open("best_feasible_candidate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nSaved: best_feasible_candidate.json")


if __name__ == "__main__":
    main()
