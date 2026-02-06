import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Open-loop model
from models.dallaman_openloop import run_simulation, extract_states

# T1DM closed-loop system
from models.dallaman_t1dm import dallaman_t1dm_ode
from simulator.closed_loop import simulate_t1dm_closed_loop
from controllers.pid import PIDController


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="GlySimLab", layout="wide")

st.title("GlySimLab")
st.subheader("Dalla Man Glucose–Insulin Simulation Platform")


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Model Selection")

model_type = st.sidebar.selectbox(
    "Choose Model",
    [
        "Open Loop (Normal Physiology)",
        "T1DM + PID Control"
    ]
)

st.sidebar.header("Simulation Settings")

sim_time = st.sidebar.slider("Simulation Time (minutes)", 100, 1000, 600, 50)
dt = st.sidebar.slider("Time Step", 0.05, 1.0, 0.1, 0.05)

st.sidebar.header("Meal Settings")

meal_size = st.sidebar.slider(
    "Meal Carbohydrate Load",
    min_value=0,
    max_value=120000,
    value=78000,
    step=1000
)

# PID parameters (only for T1DM mode)
if model_type == "T1DM + PID Control":
    st.sidebar.header("PID Parameters")

    P = st.sidebar.slider(
    "P Gain",
    min_value=0.0001,
    max_value=0.02,
    value=0.001,
    step=0.0001,
    format="%.4f"
)

I = st.sidebar.slider(
    "I Gain",
    min_value=0.000001,
    max_value=0.0005,
    value=0.00001,
    step=0.000001,
    format="%.6f"
)

D = st.sidebar.slider(
    "D Gain",
    min_value=0.0,
    max_value=0.01,
    value=0.0,
    step=0.0001,
    format="%.4f"
)

    target = st.sidebar.slider("Target Glucose", 80, 180, 140)


# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------
if st.button("Run Simulation"):

    # =====================================================
    # OPEN LOOP MODEL
    # =====================================================
    if model_type == "Open Loop (Normal Physiology)":

        sol, p, x0 = run_simulation(
            t_span=(0, sim_time),
            dt=dt,
            meal_size=meal_size
        )

        t, Gp, Gt, Ip, Ipo = extract_states(sol)

        col1, col2 = st.columns(2)

        # Glucose plot
        with col1:
            st.subheader("Glucose Dynamics")

            fig1, ax1 = plt.subplots()
            ax1.plot(t, Gp, label="Plasma Glucose (Gp)")
            ax1.plot(t, Gt, label="Tissue Glucose (Gt)")
            ax1.set_xlabel("Time (min)")
            ax1.set_ylabel("Glucose")
            ax1.legend()
            ax1.grid(True)

            st.pyplot(fig1)

        # Insulin plot
        with col2:
            st.subheader("Insulin Dynamics")

            fig2, ax2 = plt.subplots()
            ax2.plot(t, Ip, label="Plasma Insulin (Ip)")
            ax2.plot(t, Ipo, label="Portal Insulin (Ipo)")
            ax2.set_xlabel("Time (min)")
            ax2.set_ylabel("Insulin")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig2)

    # =====================================================
    # T1DM CLOSED LOOP MODEL
    # =====================================================
    else:

        # Create PID controller
        pid = PIDController(P=P, I=I, D=D, target=target)

        # Parameter set (reuse same parameters as open-loop)
        _, p, _ = run_simulation((0, 1))

        # Initial state for T1DM
        x0 = np.array([
            178.0,        # Gp
            135.0,        # Gt
            0.0,          # Il
            0.0,          # Ip
            meal_size,    # Qsto1
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,          # Isc1
            0.0           # Isc2
        ])

        t, glucose, insulin = simulate_t1dm_closed_loop(
            pid, p, x0, T=sim_time, dt=dt
        )

        col1, col2 = st.columns(2)

        # Glucose plot
        with col1:
            st.subheader("Glucose (Closed Loop)")

            fig1, ax1 = plt.subplots()
            ax1.plot(t, glucose, label="Plasma Glucose")
            ax1.axhline(target, linestyle="--", label="Target")
            ax1.set_xlabel("Time (min)")
            ax1.set_ylabel("Glucose")
            ax1.legend()
            ax1.grid(True)

            st.pyplot(fig1)

        # Insulin infusion plot
        with col2:
            st.subheader("Insulin Infusion (PID Output)")

            fig2, ax2 = plt.subplots()
            ax2.plot(t, insulin, label="Insulin Infusion Rate")
            ax2.set_xlabel("Time (min)")
            ax2.set_ylabel("Insulin Input")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig2)

        st.markdown("---")
        st.success("Closed-loop artificial pancreas simulation running.")
