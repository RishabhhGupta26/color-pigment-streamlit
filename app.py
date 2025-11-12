import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cross_decomposition import PLSRegression
import re
import warnings
import ast

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION (fixed paths)
# =============================================================================
TRAIN_REFERENCE_FILE = Path("C:/Users/risha/Project/Nestle test.xlsx")
TRAIN_SHEET = "A1B1C1E1E2_Building"

OUTPUT_DETAILS_DIR = Path("C:/Users/risha/Project2/detailed_analysis_results")
MODELS_DIR = Path("C:/Users/risha/Project2/saved_models")

# Default PLS components per output
PLS_COMPONENTS_MAP = {1: 5, 2: 6, 3: 5, 4: 6}

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title="AI Model Prediction Dashboard", layout="wide")
st.title("🔮 AI Model Prediction Dashboard (Auto Preprocessing & PLS Selection)")

st.markdown("""
This dashboard:
- Automatically loads your **training reference file** (for preprocessing)
- Lets you upload **only Testing Excel**
- Automatically applies the correct preprocessing & PLS (Yeo-Johnson + RobustScaler)
- Selects **only required feature groups** based on the chosen model
""")

# =============================================================================
# SPLIT INTO GROUPS
# =============================================================================
def split_into_groups(df):
    cols = list(df.columns)
    outputs_idx = [1, 2, 3, 49]  # outputs 1–4
    A1_1_idx = slice(4, 12)
    C1_idx = slice(12, 15)
    A1_2_idx = slice(15, 28)
    B1_idx = slice(28, 47)
    E1_idx = slice(47, 50)
    E2_idx = slice(50, 53)
    NIR_idx = slice(54, None)

    outputs_cols = [cols[i] for i in outputs_idx]
    A1_cols = cols[A1_1_idx] + cols[A1_2_idx]
    C1_cols = cols[C1_idx]
    B1_cols = cols[B1_idx]
    E1_cols = cols[E1_idx]
    E2_cols = cols[E2_idx]
    NIR_cols = cols[NIR_idx]

    groups = {
        'Outputs': df[outputs_cols],
        'A1': df[A1_cols],
        'C1': df[C1_cols],
        'B1': df[B1_cols],
        'E1': df[E1_cols],
        'E2': df[E2_cols],
        'NIR': df[NIR_cols]
    }
    return groups


# =============================================================================
# FIXED FUNCTION: PREPROCESS (RobustScaler + Yeo-Johnson + PLS)
# =============================================================================
def preprocess_groups_for_prediction(groups, reference_train, y_reference, n_components):
    processed = {}

    # ✅ 1. Non-NIR groups (same logic as training)
    for g in [n for n in groups if n not in ["NIR", "Outputs"]]:
        scaler = RobustScaler()
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_ref = reference_train[g].values
        X_test = groups[g].values
        scaler.fit(X_ref)
        pt.fit(scaler.transform(X_ref))
        processed[g] = pt.transform(scaler.transform(X_test))

    # ✅ 2. NIR group — corrected handling
    nir_train = reference_train["NIR"].values
    nir_test = groups["NIR"].values

    pls = PLSRegression(n_components=min(n_components, nir_train.shape[1], nir_train.shape[0]-1))
    pls.fit(nir_train, y_reference if y_reference.ndim == 1 else y_reference[:, 0])

    nir_train_scores = pls.transform(nir_train)
    nir_test_scores = pls.transform(nir_test)

    scaler = RobustScaler()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    scaler.fit(nir_train_scores)
    pt.fit(scaler.transform(nir_train_scores))
    nir_test_trans = pt.transform(scaler.transform(nir_test_scores))

    processed["NIR"] = nir_test_trans

    # 🧩 Optional diagnostic for scaling consistency
    if st.checkbox("📊 Show preprocessing stats"):
        for k, arr in processed.items():
            st.write(f"{k}: mean={np.mean(arr):.3f}, std={np.std(arr):.3f}")

    return processed


# =============================================================================
# LOAD TRAINING REFERENCE
# =============================================================================
def load_reference_train_data():
    if TRAIN_REFERENCE_FILE.exists():
        st.success(f"📘 Using fixed reference file: {TRAIN_REFERENCE_FILE.name}")
        ref_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)
        ref_groups = split_into_groups(ref_df)
        return ref_df, ref_groups
    else:
        st.error("❌ Reference file not found! Please place 'Nestle test.xlsx' in C:/Users/risha/Project/")
        st.stop()

train_ref_df, train_ref_groups = load_reference_train_data()

# =============================================================================
# UPLOAD TEST FILE
# =============================================================================
uploaded_file = st.file_uploader("📤 Upload your Testing Excel file", type=["xlsx"])
if not uploaded_file:
    st.stop()

xls = pd.ExcelFile(uploaded_file)
sheet_names = [s for s in xls.sheet_names if "test" in s.lower()]
selected_sheet = st.selectbox("🧪 Select Testing Sheet", sheet_names or xls.sheet_names)
test_df = pd.read_excel(xls, sheet_name=selected_sheet)
test_groups = split_into_groups(test_df)

# Verify first column consistency
if not np.array_equal(train_ref_df.iloc[:, 0].values, test_df.iloc[:, 0].values):
    st.warning("⚠️ Experiment names differ between training and testing. Ensure same ordering.")

# =============================================================================
# SELECT OUTPUT & PLS COMPONENTS
# =============================================================================
available_outputs = [1, 2, 3, 4]
selected_output = st.selectbox("🎯 Select Output to Predict", available_outputs)
default_pls = PLS_COMPONENTS_MAP[selected_output]
pls_components = st.number_input("🧩 PLS Components", value=default_pls, min_value=2, max_value=10, step=1)

# Load detailed result for this output
detail_file = OUTPUT_DETAILS_DIR / f"Output_{selected_output}_Detailed_Results.xlsx"
if not detail_file.exists():
    st.error(f"❌ Detailed file not found: {detail_file}")
    st.stop()

detail_df = pd.read_excel(detail_file, sheet_name="Top10_R2")
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
st.dataframe(detail_df[["Model", "Groups", "Test_R2", "Test_MAPE"]])

# =============================================================================
# LOAD MODEL
# =============================================================================
model_folders = list(MODELS_DIR.glob(f"Output_{selected_output}_Top10_R2")) + \
                list(MODELS_DIR.glob(f"Output_{selected_output}_Top10_MAPE"))
if not model_folders:
    st.error("❌ Model folders not found.")
    st.stop()

selected_model_folder = st.selectbox("📂 Select Model Folder", model_folders)
model_files = sorted(selected_model_folder.glob("*.pkl"))

pattern = re.compile(
    r"Rank_(\d+)_([A-Za-z0-9_]+)_R2_([0-9]+(?:\.[0-9]+)?)_MAPE_([0-9]+(?:\.[0-9]+)?)\.pkl$"
)
model_info = []
for file in model_files:
    m = pattern.search(file.name)
    if m:
        rank, model_name, r2, mape = m.groups()
        model_info.append({
            "Rank": int(rank),
            "Model": model_name,
            "R2": float(r2),
            "MAPE": float(mape),
            "File": file
        })
    else:
        st.warning(f"⚠️ Could not parse: {file.name}")

model_df = pd.DataFrame(model_info).sort_values("Rank")
st.dataframe(model_df)

selected_rank = st.selectbox("🏅 Select Model Rank", model_df["Rank"])
selected_model_row = model_df[model_df["Rank"] == selected_rank].iloc[0]
model_path = selected_model_row["File"]
model = joblib.load(model_path)
st.success(f"✅ Loaded model: {selected_model_row['Model']} (Rank {selected_rank})")

# Detect groups from detailed file
required_groups = detail_df.iloc[selected_rank - 1]["Groups"]
st.info(f"🧩 Model trained on groups: {required_groups}")

# =============================================================================
# PREPROCESS TEST DATA (only required groups)
# =============================================================================
y_train_ref = train_ref_groups["Outputs"].values[:, selected_output - 1]
processed_test = preprocess_groups_for_prediction(test_groups, train_ref_groups, y_train_ref, pls_components)
X_input = np.hstack([processed_test[g] for g in required_groups])

# =============================================================================
# PREDICT
# =============================================================================
if st.button("🚀 Predict Now"):
    try:
        y_pred = model.predict(X_input)
        y_true = test_groups["Outputs"].values[:, selected_output - 1]

        result_df = test_df.copy()
        result_df[f"Predicted_Output_{selected_output}"] = y_pred

        # Diagnostic summary
        st.write(f"🔍 **Predicted range:** min={y_pred.min():.3f}, max={y_pred.max():.3f}")
        st.write(f"📈 **True range:** min={y_true.min():.3f}, max={y_true.max():.3f}")

        st.success(f"🎯 Prediction completed using {selected_model_row['Model']} (Output {selected_output})")
        st.dataframe(result_df.head())

        st.download_button(
            label="📥 Download Predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Predicted_Output_{selected_output}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
