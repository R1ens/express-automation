from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_float_token(tok: str) -> bool:
    return bool(_FLOAT_RE.match(tok))


def _split_tokens(line: str) -> List[str]:
    return line.strip().split()


def _parse_first_numeric_block(lines: List[str], ncols: int) -> List[List[float]]:
    """
    Finds the first contiguous block of numeric rows where each row has >= ncols floats.
    Does NOT rely on headers (robust to localization / formatting / unicode issues).
    """
    rows: List[List[float]] = []
    in_block = False

    for ln in lines:
        toks = _split_tokens(ln)
        if not toks:
            if in_block:
                break
            continue

        if _is_float_token(toks[0]):
            float_toks = [t for t in toks if _is_float_token(t)]
            if len(float_toks) >= ncols:
                vals = [float(float_toks[j]) for j in range(ncols)]
                rows.append(vals)
                in_block = True
                continue

        if in_block:
            break

    return rows


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    alpha: float  # rad
    theta: float  # rad
    Jn: float  # m/s^2
    m: float  # kg
    P: float  # N
    q: float  # Pa
    Cy_alpha_traj: float  # 1/rad
    mz_alpha: float  # 1/rad
    mz_delta: float  # 1/rad


# Forces row: (t, m, P)
ForcesRow = Tuple[float, float, float]
# Position row: (t, theta, alpha, Jn)
PositionRow = Tuple[float, float, float, float]
# Additional row: (t, q, Cy_alpha_traj, mz_alpha, mz_delta)
AdditionalRow = Tuple[float, float, float, float, float]


def parse_forces_log(text: str) -> List[ForcesRow]:
    lines = text.splitlines()
    # t m m` G P X Yla`alfa Yla`delta Yla  -> 9 numeric cols
    rows = _parse_first_numeric_block(lines, ncols=9)

    out: List[ForcesRow] = []
    for r in rows:
        t = r[0]
        m = r[1]
        P = r[4]
        out.append((t, m, P))
    return out


def parse_position_log(text: str) -> List[PositionRow]:
    lines = text.splitlines()
    # t V TETA fi r x y alfa Jn -> 9 numeric cols
    rows = _parse_first_numeric_block(lines, ncols=9)

    out: List[PositionRow] = []
    for r in rows:
        t = r[0]
        theta = r[2]
        alpha = r[7]
        jn = r[8]
        out.append((t, theta, alpha, jn))
    return out


def parse_additional_log(text: str) -> List[AdditionalRow]:
    lines = text.splitlines()
    # t M q Cx Cy^alfa Cy^delta mz^alfa mz^delta delta -> 9 numeric cols
    rows = _parse_first_numeric_block(lines, ncols=9)

    out: List[AdditionalRow] = []
    for r in rows:
        t = r[0]
        q = r[2]
        cy_a = r[4]
        mz_a = r[6]
        mz_d = r[7]
        out.append((t, q, cy_a, mz_a, mz_d))
    return out


def join_logs_to_points(
        forces_rows: List[ForcesRow],
        position_rows: List[PositionRow],
        additional_rows: List[AdditionalRow],
) -> List[TrajectoryPoint]:
    """
    Robust join by row index. Truncates to the shortest list.
    """
    n = min(len(forces_rows), len(position_rows), len(additional_rows))
    points: List[TrajectoryPoint] = []

    for i in range(n):
        t_f, m, P = forces_rows[i]
        _, theta, alpha, Jn = position_rows[i]
        _, q, Cy_alpha_traj, mz_alpha, mz_delta = additional_rows[i]

        points.append(
            TrajectoryPoint(
                t=t_f,
                alpha=alpha,
                theta=theta,
                Jn=Jn,
                m=m,
                P=P,
                q=q,
                Cy_alpha_traj=Cy_alpha_traj,
                mz_alpha=mz_alpha,
                mz_delta=mz_delta,
            )
        )

    return points
