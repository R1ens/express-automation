from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, List

import domain
from calc.aerodynamics import calculate_weights, prepare_aerodynamics
from calc.ballistics import calculate_ballistics

from aero_log_parser import (
    parse_forces_log,
    parse_position_log,
    parse_additional_log,
    join_logs_to_points,
)
from aero_constraints import (
    violations_type1,
    violations_type2,
    violations_type3,
    violations_type4_points,
)
from aero_score import aggregate_score
from calc.misc import reinstall_express


def build_aerodynamics_info(
        *,
        const: Dict[str, float],
        x10_named: Dict[str, float],
) -> domain.AerodynamicsInfo:
    """
    Creates AerodynamicsInfo using project constants + the 10 optimized geometry inputs.
    """
    return domain.AerodynamicsInfo(
        L_0=const["L_0"],
        L_head=const["L_head"],
        L_stern=const["L_stern"],
        d_M=const["d_M"],
        d_stern=const["d_stern"],
        L_cm=const["L_cm"],  # if you store it, but calculate_weights ignores it anyway

        # Steering
        L_st_position=x10_named["L_st_position"],
        L_st_span=x10_named["L_st_span"],
        L_st=x10_named["L_st"],
        L_st_straight=x10_named["L_st_straight"],
        delta_st=x10_named["delta_st"],

        # Wing
        L_w_position=x10_named["L_w_position"],
        L_w_span=x10_named["L_w_span"],
        L_w=x10_named["L_w"],
        L_w_straight=x10_named["L_w_straight"],
        delta_w=x10_named["delta_w"],
    )


def evaluate_candidate(
        *,
        x10_named: Dict[str, float],
        const: Dict[str, float],
        weight_info: domain.WeightInfo,
        la_info_base: domain.LAInfo,
        targets: List[domain.TargetInfo],
        integration: domain.IntegrationInfo,
        aero_dir: str = ".",
) -> Tuple[float, Dict[str, Any]]:
    """
    Runs the full pipeline and returns (score, diagnostics).

    - Always computes type1/type2.
    - Attempts both ballistics.
    - If ballistics fails for a trajectory, assigns penalties to type3/type4 for that traj.
    """
    diag: Dict[str, Any] = {"x10": dict(x10_named)}

    # 1) AerodynamicsInfo
    ad = build_aerodynamics_info(const=const, x10_named=x10_named)
    diag["ad"] = {
        "L_st_position": ad.L_st_position,
        "L_st": ad.L_st,
        "L_w_position": ad.L_w_position,
        "L_w": ad.L_w,
    }

    # 2) Weights calculation
    wr = calculate_weights(weight_info, ad)
    diag["weights"] = {
        "m_0": wr.m_0,
        "m_k": wr.m_k,
        "L_cm": wr.L_cm,
        "S_w": wr.S_w,
        "S_st": wr.S_st,
    }

    # 3) Prepare aero file
    aero_path = Path(aero_dir) / f"candidate.ad"
    diag["aero_file"] = str(aero_path)

    # Build LAInfo for Express (engine params fixed, aero file set to generated)
    la = domain.LAInfo(
        m_0=la_info_base.m_0,
        d_middle=la_info_base.d_middle,
        mdot_A=la_info_base.mdot_A,
        mdot_B=la_info_base.mdot_B,
        t_engine=la_info_base.t_engine,
        d_nozzle=la_info_base.d_nozzle,
        J=la_info_base.J,
        V_0=la_info_base.V_0,
        theta_0=la_info_base.theta_0,
        aerodynamics=str(aero_path),
        p_nozzle=la_info_base.p_nozzle,
    )

    # 4) Constraints type1/type2 (independent of ballistics)
    v1 = violations_type1(ad, const)
    v2 = violations_type2(wr, const)

    # 5) Ballistics x2 + parse + type3/type4
    penalty_fail = float(const.get("penalty_ballistics_fail", 1e3))

    def fail_v3() -> Dict[str, float]:
        return {"c5": penalty_fail, "c6": penalty_fail, "c7": penalty_fail, "c8": penalty_fail}

    def fail_v4() -> Dict[str, float]:
        return {"c9": penalty_fail, "c10": penalty_fail, "c11": penalty_fail, "c12": penalty_fail}

    traj_results = []
    v3_list = []
    v4_list = []

    for idx, tgt in enumerate(targets):
        key = f"traj{idx + 1}"
        try:
            reinstall_express()
            prepare_aerodynamics(ad, str(aero_path))
            br = calculate_ballistics(tgt, la, integration, calculate_aerodynamics=True)
            diag[key] = {
                "m_final": br.m_final,
                "mdot_final": br.mdot_final,
                "v_final": br.v_final,
                "theta_final": br.theta_final,
                "X_final": br.X_final,
            }

            # type3
            v3 = violations_type3(br, wr, const)

            # parse logs -> points
            forces = parse_forces_log(br.trajectory_log_forces)
            pos = parse_position_log(br.trajectory_log_position)
            add = parse_additional_log(br.trajectory_log_additional)
            points = join_logs_to_points(forces, pos, add)

            diag[key]["n_points"] = len(points)

            # type4 (max over points per constraint)
            v4 = violations_type4_points(points, ad, const)

            traj_results.append(br)
            v3_list.append(v3)
            v4_list.append(v4)

        except Exception as e:
            diag[key] = {"error": str(e)}
            v3_list.append(fail_v3())
            v4_list.append(fail_v4())

    # Unpack A/B
    v3_A, v3_B = v3_list[0], v3_list[1]
    v4_A, v4_B = v4_list[0], v4_list[1]

    # 6) Aggregate into scalar score
    total, parts, detail = aggregate_score(v1, v2, v3_A, v3_B, v4_A, v4_B)

    diag["violations"] = detail
    diag["score_parts"] = parts
    diag["score_total"] = total

    return total, diag
