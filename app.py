import streamlit as st
import matplotlib.pyplot as plt

from models.dallaman_openloop import run_simulation, extract_states

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="GlySimLab - Dalla Man Simulator", layout="wide")

st.title("GlySimLab")
st.subheader("Interactive Dalla Man Glucose–Insulin Model")

st.write(
    "Simulate physiological glucose–insulin dynamics using the Dalla Man model. "
    "Adjust meal size and simulation settings to observe system response."
)

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Simulation Settings")

sim_time = st.sidebar.slider(
    "Simulation Time (minutes)",
    min_value=100,
    max_value=1000,
    value=600,
    step=50
)

dt = st.sidebar.slider(
    "Time Step (dt)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

st.sidebar.header("Meal Input")

meal_size = st.sidebar.slider(
    "Meal Carbohydrate Load (Q_sto1)",
    min_value=0,
    max_value=120000,
    value=78000,
    step=1000
)

# -------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------
if st.button("Run Simulation"):

    sol, p, x0 = run_simulation(
        t_span=(0, sim_time),
        dt=dt,
        meal_size=meal_size
    )

    t, Gp, Gt, Ip, Ipo = extract_states(sol)

    # -------------------------------------------------
    # LAYOUT
    # -------------------------------------------------
    col1, col2 = st.columns(2)

    # --------- Glucose Plot ---------
    with col1:
        st.subheader("Glucose Dynamics")

        fig1, ax1 = plt.subplots()
        ax1.plot(t, Gp, label="Plasma Glucose (Gp)")
        ax1.plot(t, Gt, label="Tissue Glucose (Gt)")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Glucose (model units)")
        ax1.legend()
        ax1.grid(True)

        st.pyplot(fig1)

    # --------- Insulin Plot ---------
    with col2:
        st.subheader("Insulin Dynamics")

        fig2, ax2 = plt.subplots()
        ax2.plot(t, Ip, label="Plasma Insulin (Ip)")
        ax2.plot(t, Ipo, label="Portal Insulin (Ipo)")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Insulin (model units)")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

    # -------------------------------------------------
    # QUICK INFO PANEL
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Simulation Info")

    st.write(f"Meal size used: {meal_size}")
    st.write(f"Simulation duration: {sim_time} minutes")
