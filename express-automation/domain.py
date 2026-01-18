from dataclasses import dataclass
import numpy as np


class ObjectNotFoundError(Exception):
    pass


@dataclass
class TargetInfo:
    velocity: float
    x: float
    y: float
    theta: float = np.pi


@dataclass
class LAInfo:
    m_0: float
    d_middle: float

    mdot_A: float
    mdot_B: float

    t_engine: float
    d_nozzle: float
    J: float

    V_0: float
    theta_0: float

    aerodynamics: str

    p_nozzle: float = 101325


@dataclass
class IntegrationInfo:
    step: float = 0.01
    output_step: float = 100


@dataclass
class BallisticsCalculationResult:
    m_final: float
    mdot_final: float
    v_final: float
    theta_final: float
    X_final: float

    trajectory_log_forces: str
    trajectory_log_position: str
    trajectory_log_additional: str


@dataclass
class WeightInfo:
    m_body: float # Without aerodynamics and fuel
    a_body: float # sum of m_i * x_cm_i without aerodynamics and fuel
    m_fuel: float
    x_cm_fuel: float # Center of mass of the engine

    rho_w: float # Density of wings material
    rho_st: float # Density of steering wings material


@dataclass
class AerodynamicsInfo:
    L_0: float
    L_head: float
    L_stern: float
    d_M: float
    d_stern: float
    L_cm: float

    # Steering
    L_st_position: float
    L_st_span: float
    L_st: float
    L_st_straight: float
    delta_st: float

    # Wing
    L_w_position: float
    L_w_span: float
    L_w: float
    L_w_straight: float
    delta_w: float

    # Mach numbers
    M: tuple[float, float, float, float, float, float, float, float, float, float] = (0.05, 0.5, 0.8, 0.9, 1.1, 1.2,
                                                                                      1.5, 2, 3, 4)
    alpha: tuple[float, float, float, float, float] = (0, 1, 2, 5, 9)

@dataclass
class WeightsCalculationResult:
    m_0: float
    m_k: float
    L_cm: float

    S_w: float
    S_st: float