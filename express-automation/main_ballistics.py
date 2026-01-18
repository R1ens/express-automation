import csv
import os
import time
from pathlib import Path
import numpy as np
import cma
import domain
from calc.ballistics import calculate_ballistics
from calc.misc import reinstall_express

# ============================================================
# Problem constants
# ============================================================

m_0 = 700.0
m_fuel = 315.0
t_max = 67.2549
m_empty = m_0 - m_fuel

d_middle = 0.4
V_0 = 375.0
J_1 = 2100.0
d_nozzle = 0.3
theta_0 = np.deg2rad(12.0)

V_target = 340.0
V_LO, V_HI = 1.5 * V_target, 2.5 * V_target

Y_min, Y_max = 900.0, 9800.0
X_1, Y_1 = Y_max / np.tan(theta_0), Y_max
X_4, Y_4 = Y_min / np.tan(theta_0), Y_min

targets = [
    domain.TargetInfo(velocity=V_target, x=X_1, y=Y_1),
    domain.TargetInfo(velocity=V_target, x=X_4, y=Y_4),
]

# ============================================================
# Search space
# x = [t_engine, J, mdot_A, kB]
# ============================================================

LOW = np.array([t_max * 0.6, J_1 * 1, 3.0, 0.8])
HIGH = np.array([t_max * 1.00, J_1 * 1, 16.0, 1.3])


def u_to_x(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return LOW + u * (HIGH - LOW)


def mdot_B_from(t_engine, mdot_A, kB):
    return 2.0 * (m_fuel - mdot_A * t_engine) / (t_engine ** 2) * kB


# ============================================================
# Violation function (core logic)
# ============================================================

def neg(x):
    return max(0.0, -x)


def vel_margin(v):
    return min(v - V_LO, V_HI - v)


def violation_for_result(r):
    return (
            neg(r.mdot_final) +
            neg(r.theta_final) +
            neg(r.m_final - m_empty) +
            neg(vel_margin(r.v_final))
    )


def evaluate(u):
    """
    Objective for CMA-ES.
    Returns (violation, diagnostics_dict)
    """
    x = u_to_x(u)
    t_engine, J, mdot_A, kB = x
    mdot_B = mdot_B_from(t_engine, mdot_A, kB)

    la = domain.LAInfo(
        m_0=m_0,
        d_middle=d_middle,
        mdot_A=float(mdot_A),
        mdot_B=float(mdot_B),
        t_engine=float(t_engine),
        d_nozzle=d_nozzle,
        J=float(J),
        V_0=V_0,
        theta_0=theta_0,
        aerodynamics='2.ad',
    )

    viols = []
    margins = []

    try:
        for tgt in targets:
            r = calculate_ballistics(tgt, la, domain.IntegrationInfo(), True)
            viols.append(violation_for_result(r))
            margins.append({
                "v_final": r.v_final,
                "vel_margin": vel_margin(r.v_final),
                "mass_margin": r.m_final - m_empty,
            })
    except Exception as e:
        return 1e9, {"error": str(e)}

    total = sum(viols)
    return total, {
        "x": x,
        "mdot_B": mdot_B,
        "t1": margins[0],
        "t2": margins[1],
    }


# ============================================================
# CSV logging
# ============================================================

LOG_ALL = Path("out/ballistics/cma_all.csv")
LOG_BEST = Path("out/ballistics/cma_best.csv")

FIELDS = [
    "gen", "eval",
    "f",
    "t_engine", "J", "mdot_A", "kB", "mdot_B",
    "t1_vel_margin", "t2_vel_margin",
    "t1_mass_margin", "t2_mass_margin",
]


def write_header(path):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writeheader()


def log_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, FIELDS).writerow(row)


# ============================================================
# CMA-ES main loop (normalized, simple)
# ============================================================

def main():
    reinstall_express()

    os.makedirs("out/ballistics")

    write_header(LOG_ALL)
    write_header(LOG_BEST)

    es = cma.CMAEvolutionStrategy(
        x0=[0.5, 0.5, 0.5, 0.5],  # start in center of [0,1]^4
        sigma0=0.25,  # 25% of range
        inopts={
            "bounds": [[0, 0, 0, 0], [1, 1, 1, 1]],
            "popsize": 24,
            "seed": 42,
            "verb_disp": 0,
        },
    )

    best_f = float("inf")
    eval_counter = 0
    t0 = time.time()
    gen = 0

    while not es.stop():
        gen += 1
        population = es.ask()
        values = []

        gen_best = float("inf")

        for u in population:
            f, info = evaluate(np.array(u))
            values.append(f)
            eval_counter += 1

            if "error" in info:
                continue

            x = info["x"]
            row = {
                "gen": gen,
                "eval": eval_counter,
                "f": f,
                "t_engine": x[0],
                "J": x[1],
                "mdot_A": x[2],
                "kB": x[3],
                "mdot_B": info["mdot_B"],
                "t1_vel_margin": info["t1"]["vel_margin"],
                "t2_vel_margin": info["t2"]["vel_margin"],
                "t1_mass_margin": info["t1"]["mass_margin"],
                "t2_mass_margin": info["t2"]["mass_margin"],
            }

            log_row(LOG_ALL, row)

            if f < best_f:
                best_f = f
                log_row(LOG_BEST, row)

            if f == 0.0:
                print("\n✅ FEASIBLE SOLUTION FOUND")
                print(row)
                return

            gen_best = min(gen_best, f)

        es.tell(population, values)

        # -------- STDOUT heartbeat --------
        elapsed = time.time() - t0
        print(
            f"[gen {gen:4d}] evals={eval_counter:5d} "
            f"gen_best={gen_best:9.4g} best={best_f:9.4g} "
            f"sigma={es.sigma:.3f} time={elapsed:.1f}s",
            flush=True,
        )

    print("\nStopped by CMA criteria.")
    print("Best violation:", best_f)


if __name__ == "__main__":
    main()
