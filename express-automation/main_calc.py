from dataclasses import dataclass
import numpy as np
import domain
import calc
from calc.ballistics import calculate_ballistics
from calc.misc import reinstall_express

reinstall_express()

# ============================================================
# INPUT DATA
# ============================================================

m_0 = 700
m_fuel = 315
t_engine = 45.48726343919475
m_empty = m_0 - m_fuel

d_middle = 0.4
V_0 = 375
J_1 = 2100
d_nozzle = 0.25
theta_0 = np.deg2rad(12)

V_target = 340

Y_min = 900
Y_max = 9800

Y_1 = Y_max
X_1 = Y_1 / np.tan(theta_0)

Y_4 = Y_min
X_4 = Y_4 / np.tan(theta_0)

mdot_A = 13.633075501711758
mdot_B = -0.32741282530564114


# ============================================================
# CHECK (DETAILED)
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


def check_detailed(r: domain.BallisticsCalculationResult) -> CheckFlags:
    return CheckFlags(
        mdot_ok=(r.mdot_final >= 0),
        theta_ok=(r.theta_final >= 0),
        mass_ok=(r.m_final >= m_empty),
        velocity_ok=(1.5 * V_target <= r.v_final <= 2.5 * V_target),
    )

# ============================================================
# CALC
# ============================================================
# t_engine: float
#     d_nozzle: float
#     J: float
#
#     V_0: float
#     theta_0: float
#
#     p_nozzle: float = 101325

target_1 = calculate_ballistics(
    target=domain.TargetInfo(
        velocity=V_target,
        x = X_1,
        y = Y_1,
    ),
    la=domain.LAInfo(
        m_0=m_0,
        d_middle=d_middle,

        mdot_A=mdot_A,
        mdot_B=mdot_B,

        t_engine=t_engine,
        d_nozzle=d_nozzle,
        J=J_1,
        V_0=V_0,
        theta_0=theta_0,
        aerodynamics='2.ad',
    ),
    integration=domain.IntegrationInfo(),
    calculate_aerodynamics=True,
)

target_4 = calculate_ballistics(
    target=domain.TargetInfo(
        velocity=V_target,
        x = X_4,
        y = Y_4,
    ),
    la=domain.LAInfo(
        m_0=m_0,
        d_middle=d_middle,

        mdot_A=mdot_A,
        mdot_B=mdot_B,

        t_engine=t_engine,
        d_nozzle=d_nozzle,
        J=J_1,
        V_0=V_0,
        theta_0=theta_0,
        aerodynamics='2.ad',
    ),
    integration=domain.IntegrationInfo(),
    calculate_aerodynamics=True,
)

print("TARGET 1")
print(target_1.trajectory_log_position)
print()
print()
print()
print()
print(
print(target_1.trajectory_log_forces))
print(check_detailed(target_1))
print()

print("TARGET 4")
print(target_4)
print(check_detailed(target_4))
print()