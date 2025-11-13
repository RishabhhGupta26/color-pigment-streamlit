# ===============================================================
# 🎨 COLOR–PIGMENT STREAMLIT APP (FORWARD + INVERSE + ΔE CHECK)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, base64

# Handle numpy.asscalar deprecation (for colormath compatibility)
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: np.asarray(x).item()

# Try importing advanced color difference function
try:
    from colormath.color_objects import LabColor
    from colormath.color_diff import delta_e_cie2000
    HAVE_CIE2000 = True
except Exception:
    HAVE_CIE2000 = False


# ===============================================================
# 🖼️ CLEAN HEADER WITH RIGHT-ALIGNED LOGO
# ===============================================================

st.set_page_config(page_title="🎨 Color–Pigment Predictor", layout="wide", page_icon="🎨")

# convert image to base64
def get_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

logo_file = "logo_off.png"   # << USE THIS LOGO


# --- CSS: title stays in one line + logo on right ---
st.markdown("""
<style>
.header-clean {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 5px 20px 5px;
}
.header-clean h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    white-space: nowrap;
}
.header-clean img {
    width: 140px;
    height: auto;
}
</style>
""", unsafe_allow_html=True)


# --- Header layout ---
if os.path.exists(logo_file):
    logo_base64 = get_base64(logo_file)
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
    st.warning("⚠️ logo_off.png not found!")

st.markdown("Use trained models to convert between **Pigment → LAB** and **LAB → Pigment + LAB + ΔE**.")


# ===============================================================
# 2️⃣ LOAD MODELS
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
# 3️⃣ DEFINE COLUMNS
# ===============================================================

pigment_columns = [
    "Base_P","Base_C","Base_D","Base_M","Black","Brown","Red (IOR)","IOY",
    "Light Yellow","Blue","Green","Red","Yellow","Orange","Violet","Maroon"
]
lab_cols = ["L", "a", "B"]


# ===============================================================
# 4️⃣ SIDEBAR
# ===============================================================

st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio("Choose prediction mode:", ["Forward: Pigments → LAB", "Inverse: LAB → Pigments + LAB + ΔE"])


# ===============================================================
# 5️⃣ FORWARD MODEL
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
        st.dataframe(input_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### 🎨 Predicted LAB Values")
        lab_df = pd.DataFrame([prediction], columns=lab_cols)
        st.dataframe(lab_df.style.format("{:.3f}"), use_container_width=True)


# ===============================================================
# 6️⃣ INVERSE MODEL
# ===============================================================

elif mode == "Inverse: LAB → Pigments + LAB + ΔE":
    st.subheader("🎯 Inverse Model — Predict Pigments from LAB and Validate via Forward Model")

    col1, col2, col3 = st.columns(3)
    with col1:
        L_val = st.number_input("L", 0.0, 100.0, 90.0)
    with col2:
        a_val = st.number_input("a", -128.0, 127.0, -0.60)
    with col3:
        B_val = st.number_input("B", -128.0, 127.0, 1.5)

    if st.button("🎨 Predict Pigments + LAB + ΔE"):
        lab_input_df = pd.DataFrame([[L_val, a_val, B_val]], columns=lab_cols)
        predicted_pigments = inverse_model.predict(lab_input_df)
        pigment_df = pd.DataFrame(predicted_pigments, columns=pigment_columns)

        reconstructed_lab = forward_model.predict(pigment_df)
        reconstructed_lab_df = pd.DataFrame(reconstructed_lab, columns=lab_cols)

        target = lab_input_df.iloc[0].values
        recon  = reconstructed_lab_df.iloc[0].values

        # ΔE calculation
        try:
            delta_e = float(delta_e_cie2000(LabColor(*target), LabColor(*recon)))
        except:
            delta_e = float(np.linalg.norm(target - recon))

        delta_df = pd.DataFrame(
            [np.abs(recon - target)],
            columns=["ΔL", "Δa", "ΔB"]
        )
        delta_df["ΔE"] = delta_e

        st.markdown("#### 🎨 Predicted Pigment Composition")
        st.dataframe(pigment_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### 🔁 Reconstructed LAB")
        st.dataframe(reconstructed_lab_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### 📏 LAB Difference + ΔE")
        st.dataframe(delta_df.style.format("{:.3f}"), use_container_width=True)

        # Interpretation
        if delta_e < 1:
            note = "🟢 Excellent color match (ΔE < 1)"
        elif delta_e < 3:
            note = "🟡 Good match (ΔE < 3)"
        elif delta_e < 5:
            note = "🟠 Acceptable (ΔE < 5)"
        else:
            note = "🔴 Large difference (ΔE ≥ 5)"

        st.markdown(f"**Color difference interpretation:** {note}")


# ===============================================================
# 7️⃣ SIDEBAR INFO
# ===============================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**App Features**
- Forward Model → LAB prediction  
- Inverse Model → Pigment + LAB + ΔE  
- Uses CIEDE2000 (ΔE) when available  
""")
