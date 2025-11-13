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
# 🖼️ CLEAN CLIENT LOGO (LEFT ALIGNED) + PAGE CONFIGURATION
# ===============================================================

st.set_page_config(page_title="🎨 Color–Pigment Predictor", layout="wide", page_icon="🎨")

# --- Convert image to base64 string ---
def get_base64(logo_path):
    with open(logo_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# --- CSS for small left-aligned logo ---
st.markdown("""
    <style>
        .header-container {
            display: flex;
            align-items: center;
            padding: 10px 0 20px 0;
        }
        .header-container img {
            width: 120px;  
            margin-right: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Display the transparent logo ---
logo_path = "logo_transparent.png"   # UPDATED file name

if os.path.exists(logo_path):
    logo_base64 = get_base64(logo_path)
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}">
            <h1 style="margin: 0; padding-top: 10px;">🎨 Bidirectional Color–Pigment Prediction System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("🎨 Bidirectional Color–Pigment Prediction System")
    st.warning("⚠️ logo_transparent.png not found! Please add it next to app.py.")


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
# 4️⃣ SIDEBAR MODE
# ===============================================================
st.sidebar.header("⚙️ Mode Selection")
mode = st.sidebar.radio("Choose prediction mode:", ["Forward: Pigments → LAB", "Inverse: LAB → Pigments + LAB + ΔE"])


# ===============================================================
# 5️⃣ FORWARD MODEL MODE
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
# 6️⃣ INVERSE MODEL MODE + ΔE CALCULATION
# ===============================================================
elif mode == "Inverse: LAB → Pigments + LAB + ΔE":
    st.subheader("🎯 Inverse Model — Predict Pigments from LAB and Validate via Forward Model")

    st.markdown("##### Enter Target LAB Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        L_val = st.number_input("L", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
    with col2:
        a_val = st.number_input("a", min_value=-128.0, max_value=127.0, value=-0.60, step=0.1)
    with col3:
        B_val = st.number_input("B", min_value=-128.0, max_value=127.0, value=1.5, step=0.1)

    if st.button("🎨 Predict Pigments + LAB + ΔE"):
        lab_input_df = pd.DataFrame([[L_val, a_val, B_val]], columns=lab_cols)
        predicted_pigments = inverse_model.predict(lab_input_df)
        pigment_df = pd.DataFrame(predicted_pigments, columns=pigment_columns)

        reconstructed_lab = forward_model.predict(pigment_df)
        reconstructed_lab_df = pd.DataFrame(reconstructed_lab, columns=lab_cols)

        target_LAB = lab_input_df.iloc[0].values
        recon_LAB = reconstructed_lab_df.iloc[0].values

        def safe_delta_e_cie2000(c1, c2):
            try:
                return float(delta_e_cie2000(c1, c2))
            except:
                return float(np.linalg.norm(np.array([
                    c1.lab_l - c2.lab_l,
                    c1.lab_a - c2.lab_a,
                    c1.lab_b - c2.lab_b
                ])))

        if HAVE_CIE2000:
            color1 = LabColor(*target_LAB)
            color2 = LabColor(*recon_LAB)
            delta_e = safe_delta_e_cie2000(color1, color2)
        else:
            delta_e = np.linalg.norm(target_LAB - recon_LAB)

        delta_components = np.abs(reconstructed_lab_df.values - lab_input_df.values)
        delta_df = pd.DataFrame(delta_components, columns=["ΔL", "Δa", "ΔB"])
        delta_df["ΔE"] = [delta_e]

        st.markdown("#### 🎨 Predicted Pigment Composition")
        st.dataframe(pigment_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### 🔁 Reconstructed LAB (via Forward Model)")
        st.dataframe(reconstructed_lab_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### 📏 LAB Difference + ΔE")
        st.dataframe(delta_df.style.format("{:.3f}"), use_container_width=True)

        if delta_e < 1:
            note = "🟢 Excellent color match (ΔE < 1 — visually identical)"
        elif delta_e < 3:
            note = "🟡 Good match (ΔE < 3 — small visible difference)"
        elif delta_e < 5:
            note = "🟠 Acceptable (ΔE < 5 — noticeable but OK)"
        else:
            note = "🔴 Large difference (ΔE ≥ 5 — visually distinct)"

        st.markdown(f"**Color difference interpretation:** {note}")


# ===============================================================
# 7️⃣ SIDEBAR INFO
# ===============================================================
st.sidebar.markdown("---")
st.sidebar.info("""
**App Features**
- Forward Model → LAB prediction  
- Inverse Model → Pigment + LAB + ΔE  
- ΔE uses CIEDE2000 if available (fallback to Euclidean)  
""")
