# ===============================================================
# 🎨 COLOR–PIGMENT STREAMLIT APP (FORWARD + INVERSE + ΔE CHECK)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, base64
from PIL import Image

# ===============================================================
# MUST be the first Streamlit command
# ===============================================================
st.set_page_config(page_title="🎨 Color–Pigment Predictor", layout="wide", page_icon="🎨")


# ===============================================================
# Load logo safely + convert to base64
# ===============================================================

def load_logo_base64(path):
    try:
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except:
        return None

LOGO_FILE = "logo_on.png"   # make sure this exact file exists

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
    st.warning(f"⚠️ Logo file '{LOGO_FILE}' not found!")

st.markdown("Use trained models to convert between **Pigment → LAB** and **LAB → Pigment + LAB + ΔE**.")

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
mode = st.sidebar.radio("Choose prediction mode:", ["Forward: Pigments → LAB", "Inverse: LAB → Pigments + LAB + ΔE"])

# ===============================================================
# FORWARD MODEL
# ===============================================================

if mode == "Forward: Pigments → LAB":
    st.subheader("🎯 Forward Model — Predict LAB from Pigments")
    st.markdown("##### Enter Pigment Composition (phr or %)")

    cols = st.columns(4)
    input_data = {}
    for i, col in enumerate(pigment_columns):
        with cols[i % 4]:
            input_data[col] = st.number_input(col, min_value=0.0, value=0.0, step=0.1)

    if st.button("🔮 Predict LAB"):
        input_df = pd.DataFrame([input_data])
        prediction = forward_model.predict(input_df)[0]

        st.markdown("#### 🧾 Input Pigment Composition")
        st.dataframe(input_df, use_container_width=True)

        st.markdown("#### 🎨 Predicted LAB Values")
        lab_df = pd.DataFrame([prediction], columns=lab_cols)
        st.dataframe(lab_df, use_container_width=True)

# ===============================================================
# INVERSE MODEL
# ===============================================================

elif mode == "Inverse: LAB → Pigments + LAB + ΔE":
    st.subheader("🎯 Inverse Model — Predict Pigments from LAB and Validate via Forward Model")

    col1, col2, col3 = st.columns(3)
    with col1:
        L_val = st.number_input("L", 0.0, 100.0, 91.93)
    with col2:
        a_val = st.number_input("a", -128.0, 127.0, -0.74)
    with col3:
        B_val = st.number_input("B", -128.0, 127.0, 1.94)

    if st.button("🎨 Predict Pigments + Lab + ΔE"):
        # ---------- Step 1: Target LAB ----------
        lab_input_df = pd.DataFrame([[L_val, a_val, B_val]], columns=lab_cols)

        # ---------- Step 2: Predict pigments ----------
        predicted_pigments = inverse_model.predict(lab_input_df)
        pigment_df = pd.DataFrame(predicted_pigments, columns=pigment_columns)

        # ---------- Step 3: Reconstruct LAB using forward model ----------
        reconstructed_lab = forward_model.predict(pigment_df)[0]

        # ---------- Step 4: Convert safely to numeric arrays ----------
        target_arr = lab_input_df.iloc[0].astype(float).values
        recon_arr  = np.array(reconstructed_lab, dtype=float)

        # ---------- Step 5: ΔE Calculation ----------
        try:
            delta_e = float(delta_e_cie2000(LabColor(*target_arr), LabColor(*recon_arr)))
        except:
            delta_e = float(np.linalg.norm(target_arr - recon_arr))

        # ---------- Step 6: Component-wise ΔL / Δa / ΔB ----------
        deltas = np.abs(recon_arr - target_arr)

        delta_df = pd.DataFrame(
            [deltas],
            columns=["ΔL", "Δa", "ΔB"]
        )
        delta_df["ΔE"] = delta_e

        # ---------- DISPLAY ----------
        st.markdown("#### 🎨 Predicted Pigment Composition")
        st.dataframe(pigment_df, use_container_width=True)

        st.markdown("#### 🔁 Reconstructed LAB")
        st.dataframe(pd.DataFrame([reconstructed_lab], columns=lab_cols), use_container_width=True)

        st.markdown("#### 📏 LAB Difference + Lab + ΔE")
        st.dataframe(delta_df, use_container_width=True)
