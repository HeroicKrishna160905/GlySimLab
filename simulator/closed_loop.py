import numpy as np
from models.dallaman_t1dm import dallaman_t1dm_ode


# -----------------------------
# Fake observation for PID
# -----------------------------
class Observation:
    def __init__(self, glucose):
        self.CGM = glucose


# -----------------------------
# RK4 integrator step
# -----------------------------
def rk4_step(f, t, x, dt, p, u):
    k1 = f(t, x, p, u)
    k2 = f(t + dt/2, x + dt/2 * k1, p, u)
    k3 = f(t + dt/2, x + dt/2 * k2, p, u)
    k4 = f(t + dt, x + dt * k3, p, u)

    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# CLOSED LOOP SIMULATION
# -----------------------------
def simulate_t1dm_closed_loop(pid, p, x0, T=600, dt=0.1):
    """
    Closed-loop simulation using RK4.
    PID controls insulin infusion.
    """

    time = []
    glucose = []
    insulin = []

    x = x0.copy()
    t = 0

    while t <= T:

        # Current plasma glucose
        Gp = x[0]

        # Create observation for PID
        obs = Observation(Gp)

        action = pid.policy(
            obs,
            reward=None,
            done=False,
            sample_time=dt
        )

        # PID output = insulin infusion rate
        u = action.basal

        # Physiological constraint:
        # No negative insulin
        u = max(0, u)

        # RK4 step
        x = rk4_step(dallaman_t1dm_ode, t, x, dt, p, u)

        # Store data
        time.append(t)
        glucose.append(Gp)
        insulin.append(u)

        t += dt

    return np.array(time), np.array(glucose), np.array(insulin)
