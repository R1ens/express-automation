from calc.const import app_dir
from domain import WeightInfo, AerodynamicsInfo, WeightsCalculationResult


# calculate_weights calculates the weight-parameters using given WeightInfo and AerodynamicsInfo (ignoring L_cm in AerodynamicsInfo)
def calculate_weights(w: WeightInfo, ad: AerodynamicsInfo) -> WeightsCalculationResult:
    # Surface area of wings and steering
    S_w = ((ad.L_w_span - ad.d_M) / 2 *
           (ad.L_w_straight + (ad.L_w - ad.L_w_straight) / 2))
    S_st = ((ad.L_st_span - ad.d_M) / 2 *
            (ad.L_st_straight + (ad.L_st - ad.L_st_straight) / 2))

    # Mass of wings and steering
    m_w = 4 * S_w * ad.delta_w * w.rho_w
    m_st = 4 * S_st * ad.delta_st * w.rho_st
    m_ad = m_w + m_st

    # Center of mass of wings and steering
    L_w_triangle = ad.L_w - ad.L_w_straight
    x_cg_w_rel = (0.5 * ad.L_w_straight ** 2 + 0.5 * ad.L_w_straight * L_w_triangle + (1 / 6) * L_w_triangle ** 2) / (
            ad.L_w_straight + 0.5 * L_w_triangle)
    x_cg_w_abs = ad.L_w_position + x_cg_w_rel

    L_st_triangle = ad.L_st - ad.L_st_straight
    x_cg_st_rel = (0.5 * ad.L_st_straight ** 2 + 0.5 * ad.L_st_straight * L_st_triangle + (
            1 / 6) * L_st_triangle ** 2) / (
                          ad.L_st_straight + 0.5 * L_st_triangle)
    x_cg_st_abs = ad.L_st_position + x_cg_st_rel

    # Center of mass calculations
    m_k = w.m_body + m_ad
    m_0 = m_k + w.m_fuel

    a_k = w.a_body + m_w * x_cg_w_abs + m_st * x_cg_st_abs
    a_0 = a_k + w.m_fuel * w.x_cm_fuel

    x_cm_k = a_k / m_k
    x_cm_0 = a_0 / m_0
    x_cm = (x_cm_k + x_cm_0) / 2

    return WeightsCalculationResult(
        m_0=m_0,
        m_k=m_k,
        L_cm=x_cm,
        S_w=S_w,
        S_st=S_st,
    )


# prepare_aerodynamics prepares aerodynamic params file (filename.ad) for Express using given input variables
def prepare_aerodynamics(ad: AerodynamicsInfo, output: str):
    params = [
        ad.L_0 - ad.L_stern,
        ad.delta_w,
        ad.L_w_position,
        ad.L_cm,
        ad.d_M,

        ad.d_stern,
        ad.L_w_span,
        ad.L_st_position,
        ad.delta_st,
        ad.L_st_straight,

        ad.L_st_span,
        ad.L_w_straight,
        ad.L_head,
        ad.L_st,
        ad.L_w,

        ad.L_0,

        *ad.M,
        *ad.alpha,
    ]

    with open(app_dir + "\\" + output, 'w') as f:
        for i in range(len(params)):
            f.write(f"{params[i]:.15f}    ad[{i}]\n")
