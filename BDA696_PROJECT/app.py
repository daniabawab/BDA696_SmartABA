import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

ART = Path("artifacts")
MODEL_KERAS = ART / "aba_success_model.keras"
PIPE_PATH = ART / "preprocessor.joblib"
INTV_PATH = ART / "interventions.txt"
LOGO_PATH = "logo.png"

st.set_page_config(page_title="SMART ABA — Your At-Home Therapist", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background-color: #FFF7E8;
        }
        h1 {
            color: #0A3A5A !important;
            font-weight: 900 !important;
        }
        .stSubheader {
            color: #0A3A5A !important;
        }
        .stButton>button {
            background-color: #F47A42 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.2rem !important;
            border: none !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }
        .stDataFrame {
            border: 3px solid #0A3A5A !important;
            border-radius: 10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=300)

st.title("SMART ABA — Your At-Home Therapist")

assert MODEL_KERAS.exists(), f"Missing model at {MODEL_KERAS}"
assert PIPE_PATH.exists(), f"Missing preprocessor at {PIPE_PATH}"
assert INTV_PATH.exists(), f"Missing interventions at {INTV_PATH}"

model = tf.keras.models.load_model(str(MODEL_KERAS))
preproc = joblib.load(PIPE_PATH)

with open(INTV_PATH) as f:
    ALL_INTERVENTIONS = [x.strip() for x in f if x.strip()]

CONFIG = {
    "rename": {
        "behavior_type": "Behavior Type",
        "behavior_intensity": "Behavior Intensity",
        "setting": "Setting",
        "parent_stress_level": "Parent Stress Level",
        "diagnosis_level": "Diagnosis Level",
        "behavior_frequency_last_week": "Behavior Frequency (Last Week)"
    },
    "include": {
        "categorical": ["behavior_type", "setting", "diagnosis_level"],
        "numeric": ["behavior_intensity", "parent_stress_level", "behavior_frequency_last_week"]
    },
    "options": {
        "behavior_type": ["Non_compliance", "Tantrum", "Aggression", "SIB", "Elopement", "Other"],
        "setting": ["Home", "School", "Community", "Clinic"],
        "diagnosis_level": ["Level_1", "Level_2", "Level_3"]
    },
    "defaults": {
        "behavior_intensity": 2,
        "parent_stress_level": 4,
        "behavior_frequency_last_week": 7
    },
    "enable_other_option": True
}

def pretty(col):
    return CONFIG["rename"].get(col, col.replace("_", " ").title())

cat_features = CONFIG["include"]["categorical"]
num_features = CONFIG["include"]["numeric"]

st.subheader("Fill in the fields below to get your personalized Top-3 intervention recommendations.")

context = {}
cols = st.columns(3)
i = 0

for c in cat_features:
    with cols[i % 3]:
        options = CONFIG["options"].get(c, ["Unknown"])
        if CONFIG["enable_other_option"]:
            options = options + ["Other…"]
            choice = st.selectbox(pretty(c), options, index=0)
            if choice == "Other…":
                custom_value = st.text_input(f"Enter custom value for {pretty(c)}")
                context[c] = custom_value.strip() if custom_value.strip() else options[0]
            else:
                context[c] = choice
        else:
            context[c] = st.selectbox(pretty(c), options)
    i += 1

def default_num(col):
    return float(CONFIG["defaults"].get(col, 0.0))

for c in num_features:
    with cols[i % 3]:
        context[c] = st.number_input(pretty(c), value=default_num(c))
    i += 1

if st.button("Recommend Top-3 Interventions"):
    rows = []
    for itv in ALL_INTERVENTIONS:
        row = dict(context)
        row["intervention"] = itv
        rows.append(row)

    cand_df = pd.DataFrame(rows)

    for c in num_features:
        cand_df[c] = pd.to_numeric(cand_df[c], errors="coerce").fillna(0.0)

    trained_cat = preproc.transformers_[0][2]
    trained_num = preproc.transformers_[1][2]

    for col in preproc.feature_names_in_:
        if col not in cand_df.columns:
            if col in trained_cat:
                cand_df[col] = "Unknown"
            elif col in trained_num:
                cand_df[col] = 0
            else:
                cand_df[col] = 0

    X_cand = preproc.transform(cand_df).astype("float32")
    probs = model.predict(X_cand, verbose=0).ravel()

    out = pd.DataFrame({
        "Intervention": cand_df["intervention"],
        "Predicted Success": probs
    }).sort_values("Predicted Success", ascending=False).head(3).reset_index(drop=True)

    st.success("Top-3 Recommendations")
    st.dataframe(out, use_container_width=True)
