# ===============================================================
# 🎨 COLOR–PIGMENT STREAMLIT APP (FORWARD + INVERSE + ΔE CHECK)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, base64
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# ===============================================================
# MUST be the first Streamlit command
# ===============================================================
st.set_page_config(page_title="🎨 Color–Pigment Predictor", layout="wide", page_icon="🎨")

# ===============================================================
# FIX for NumPy deprecated asscalar()
# ===============================================================
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: x.item()

# ===============================================================
# 🔐 LOGIN / PASSWORD AUTHENTICATION
# ===============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h3 style='text-align:center;'>🔐 Secure Login</h3>", unsafe_allow_html=True)
    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if password == "color@123":
            st.session_state.logged_in = True
            st.success("✔ Login successful!")
        else:
            st.error("❌ Incorrect password")

if not st.session_state.logged_in:
    login()
    st.stop()

# ===============================================================
# Initialize session state variables
# ===============================================================
if "forward_lab" not in st.session_state:
    st.session_state.forward_lab = None

if "forward_delta" not in st.session_state:
    st.session_state.forward_delta = None

if "inverse_pigments" not in st.session_state:
    st.session_state.inverse_pigments = None

if "inverse_delta" not in st.session_state:
    st.session_state.inverse_delta = None

# ===============================================================
# Load logo safely + convert to base64
# ===============================================================

def load_logo_base64(path):
    try:
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except:
        return None

LOGO_FILE = "logo_on.png"
logo_base64 = load_logo_base64(LOGO_FILE)

# ===============================================================
# HEADER CSS
# ===============================================================
st.markdown("""
<style>
.header-clean {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px 20px 10px;
}
.header-clean h1 {
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    white-space: nowrap;
}
.header-clean img {
    width: 130px;
    height: auto;
}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# HEADER BLOCK
# ===============================================================
if logo_base64:
    st.markdown(
        f"""
        <div class="header-clean">
            <h1>🎨 Bidirectional Color–Pigment Prediction System</h1>
            <img src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("🎨 Bidirectional Color–Pigment Prediction System")

st.markdown("Use trained models to convert between **Pigment → LAB** and **LAB → Pigments**.")

# ===============================================================
# LOAD MODELS
# ===============================================================
FORWARD_MODEL_PATH = "ForwardModel_RF.joblib"
INVERSE_MODEL_PATH = "InverseModel_RF.joblib"

if not (os.path.exists(FORWARD_MODEL_PATH) and os.path.exists(INVERSE_MODEL_PATH)):
    st.error("❌ Model files not found! Ensure both .joblib files are in the same folder as app.py.")
    st.stop()

forward_model = joblib.load(FORWARD_MODEL_PATH)
inverse_model = joblib.load(INVERSE_MODEL_PATH)
st.success("✅ Models loaded successfully!")

# ===============================================================
# COLUMN DEFINITIONS
# ===============================================================
pigment_columns = [
    "Base_P","Base_C","Base_D","Base_M","Black","Brown","Red (IOR)","IOY",
    "Light Yellow","Blue","Green","Red","Yellow","Orange","Violet","Maroon"
]
lab_cols = ["L", "a", "B"]

# ===============================================================
# SIDEBAR
# ===============================================================
st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio("Choose prediction mode:", ["Forward: Pigments → LAB", "Inverse: LAB → Pigments"])

# ΔE2000 Information
st.sidebar.markdown("""
### 📏 ΔE2000 Color Difference  
**Formula Concept:**  
ΔE2000 = difference between two LAB colors considering  
lightness, chroma & hue corrections.

**Meaning of LAB:**  
- **L*** → Lightness  
- **a*** → Green (−) to Red (+)  
- **b*** → Blue (−) to Yellow (+)

### 🎯 Interpretation
- **ΔE < 1:** Excellent  
- **1–2:** Very good  
- **2–5:** Acceptable  
- **> 5:** Poor  
""")

# ===============================================================
# FORWARD MODEL
# ===============================================================

if mode == "Forward: Pigments → LAB":
    st.subheader("🎯 Forward Model — Predict LAB from Pigments")

    cols = st.columns(4)
    input_data = {}

    for i, col in enumerate(pigment_columns):
        with cols[i % 4]:
            input_data[col] = st.number_input(col, min_value=0.0, value=0.0, step=0.1)

    # BUTTON → STORE RESULT IN SESSION STATE
    if st.button("🔮 Predict LAB"):
        input_df = pd.DataFrame([input_data])
        st.session_state.forward_lab = forward_model.predict(input_df)[0]

    # SHOW PERSISTENT PREDICTION
    if st.session_state.forward_lab is not None:
        st.markdown("#### 🎨 Predicted LAB Values")
        st.dataframe(pd.DataFrame([st.session_state.forward_lab], columns=lab_cols), use_container_width=True)

        # Actual Lab for DeltaE
        st.markdown("### 📌 Compare With Actual LAB (Optional)")
        c1, c2, c3 = st.columns(3)
        act_L = c1.number_input("Actual L", 0.0, 100.0, 0.0)
        act_a = c2.number_input("Actual a", -128.0, 127.0, 0.0)
        act_B = c3.number_input("Actual B", -128.0, 127.0, 0.0)

        if st.button("📏 Compute ΔE2000"):
            pred = st.session_state.forward_lab
            deltaE = delta_e_cie2000(LabColor(*pred), LabColor(act_L, act_a, act_B))

            deltas = np.abs(np.array([pred[0]-act_L, pred[1]-act_a, pred[2]-act_B]))

            st.session_state.forward_delta = pd.DataFrame([{
                "ΔL": deltas[0],
                "Δa": deltas[1],
                "ΔB": deltas[2],
                "ΔE2000": deltaE
            }])

        if st.session_state.forward_delta is not None:
            st.markdown("#### 📏 LAB Difference + ΔE2000")
            st.dataframe(st.session_state.forward_delta, use_container_width=True)

# ===============================================================
# INVERSE MODEL
# ===============================================================

elif mode == "Inverse: LAB → Pigments":
    st.subheader("🎯 Inverse Model — Predict Pigments from LAB")

    c1, c2, c3 = st.columns(3)
    L_val = c1.number_input("Target L", 0.0, 100.0, 91.93)
    a_val = c2.number_input("Target a", -128.0, 127.0, -0.74)
    B_val = c3.number_input("Target B", -128.0, 127.0, 1.94)

    if st.button("🎨 Predict Pigments"):
        target_lab_df = pd.DataFrame([[L_val, a_val, B_val]], columns=lab_cols)
        st.session_state.inverse_pigments = inverse_model.predict(target_lab_df)

    if st.session_state.inverse_pigments is not None:
        pigment_df = pd.DataFrame(st.session_state.inverse_pigments, columns=pigment_columns)
        st.markdown("#### 🎨 Suggested Pigment Composition")
        st.dataframe(pigment_df, use_container_width=True)

        st.markdown("### 📌 Measured LAB From These Pigments")
        m1, m2, m3 = st.columns(3)
        meas_L = m1.number_input("Measured L", 0.0, 100.0, 0.0)
        meas_a = m2.number_input("Measured a", -128.0, 127.0, 0.0)
        meas_B = m3.number_input("Measured B", -128.0, 127.0, 0.0)

        if st.button("📏 Compute ΔE2000 Difference"):
            target = LabColor(L_val, a_val, B_val)
            measured = LabColor(meas_L, meas_a, meas_B)

            deltaE = delta_e_cie2000(target, measured)
            deltas = np.abs(np.array([L_val-meas_L, a_val-meas_a, B_val-meas_B]))

            st.session_state.inverse_delta = pd.DataFrame([{
                "ΔL": deltas[0],
                "Δa": deltas[1],
                "ΔB": deltas[2],
                "ΔE2000": deltaE
            }])

        if st.session_state.inverse_delta is not None:
            st.markdown("#### 📏 LAB Difference + ΔE2000")
            st.dataframe(st.session_state.inverse_delta, use_container_width=True)
