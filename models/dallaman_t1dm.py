import numpy as np
from scipy.integrate import solve_ivp


def dallaman_t1dm_ode(t, x, p, u=0):
    """
    T1DM version:
    - No endogenous insulin secretion
    - External insulin infusion u(t)
    """

    # State vector
    # [Gp, Gt, Il, Ip, Qsto1, Qsto2, Qgut, I1, Id, X, Isc1, Isc2]
    Gp, Gt, Il, Ip, Q_sto1, Q_sto2, Q_gut, I1, Id, X, Isc1, Isc2 = x

    # Parameters
    V_G = p['V_G']
    k_1 = p['k_1']
    k_2 = p['k_2']
    G_b = p['G_b']
    V_I = p['V_I']
    m_1 = p['m_1']
    m_2 = p['m_2']
    m_4 = p['m_4']
    k_abs = p['k_abs']
    k_gri = p['k_gri']
    f = p['f']
    BW = p['BW']
    k_p1 = p['k_p1']
    k_p2 = p['k_p2']
    k_p3 = p['k_p3']
    k_p4 = p['k_p4']
    k_i = p['k_i']
    U_ii = p['U_ii']
    V_m0 = p['V_m0']
    V_mX = p['V_mX']
    K_m0 = p['K_m0']
    p_2U = p['p_2U']
    part = p['part']

    # Subcutaneous insulin parameters
    ka1 = 0.01
    ka2 = 0.01
    kd = 0.01

    # --- Glucose production ---
    EGP = k_p1 - k_p2 * Gp - k_p3 * Id - k_p4 * Ip

    V_mmax = (1 - part) * (V_m0 + V_mX * X)
    U_id = V_mmax * Gt / (K_m0 + Gt)

    Ra = f * k_abs * Q_gut / BW

    # --- Glucose dynamics ---
    dGp = EGP + Ra - U_ii - k_1 * Gp + k_2 * Gt
    dGt = -U_id + k_1 * Gp - k_2 * Gt

    # --- Insulin kinetics ---
    dIl = (-m_1) * Il + m_2 * Ip
    dIp = (-m_2) * Ip - m_4 * Ip + m_1 * Il + ka1 * Isc1 + ka2 * Isc2

    # --- GI system ---
    dQsto1 = -k_gri * Q_sto1
    dQsto2 = k_gri * Q_sto1 - 0.05 * Q_sto2
    dQgut = -k_abs * Q_gut + 0.05 * Q_sto2

    # --- Insulin action ---
    I = Ip / V_I
    dI1 = -k_i * (I1 - I)
    dId = -k_i * (Id - I1)
    dX = -p_2U * X + p_2U * I

    # --- Subcutaneous insulin compartments ---
    dIsc1 = u - (ka1 + kd) * Isc1
    dIsc2 = kd * Isc1 - ka2 * Isc2

    return np.array([
        dGp, dGt, dIl, dIp,
        dQsto1, dQsto2, dQgut,
        dI1, dId, dX,
        dIsc1, dIsc2
    ])
