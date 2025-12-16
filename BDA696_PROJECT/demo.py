import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

CSV_PATH = "/Users/daniabawab/Desktop/BDA696_PROJECT/data/ABA_Sessions_with_TimeAndDay.csv"  
df = pd.read_csv(CSV_PATH)

print("Data shape:", df.shape)
print("Columns:", list(df.columns))

target_col = "outcome_success"

cat_cols = [
    "gender",
    "diagnosis_level",
    "setting",
    "antecedent",
    "behavior_type",
    "reinforcement",
    "parent_training_level",
    "medication_use",
    "school_support",
    "environment_noise_level",
    "reward_preference",
    "intervention",          
]

num_cols = [
    "age",
    "behavior_intensity",
    "behavior_duration_min",
    "sibling_count",
    "therapy_hours_week",
    "behavior_frequency_last_week",
    "parent_stress_level",
    "session_hour",
]
drop_cols = [
    "day_of_week", "Description", "timestamp", "session_time", "event_id"
]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

keep_cols = [c for c in cat_cols + num_cols + [target_col] if c in df.columns]
df = df[keep_cols].copy()

df = df.dropna(subset=[target_col])
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

df[target_col] = df[target_col].astype(int)

ALL_INTERVENTIONS = sorted(df["intervention"].dropna().unique().tolist())
print("Interventions ({}):".format(len(ALL_INTERVENTIONS)), ALL_INTERVENTIONS[:10], "..." if len(ALL_INTERVENTIONS) > 10 else "")

X = df.drop(columns=[target_col])
y = df[target_col].values


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

categorical_features = [c for c in cat_cols if c in X.columns]
numeric_features = [c for c in num_cols if c in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

preprocessor.fit(X_train)
X_train_np = preprocessor.transform(X_train)
X_val_np = preprocessor.transform(X_val)
X_test_np = preprocessor.transform(X_test)

input_dim = X_train_np.shape[1]
print("Input dimension after encoding:", input_dim)

def make_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")  
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model

model = make_model(input_dim)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
   
]

history = model.fit(
    X_train_np, y_train,
    validation_data=(X_val_np, y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_auc, test_acc = model.evaluate(X_test_np, y_test, verbose=0)
print(f"Test AUC: {test_auc:.3f} | Test Acc: {test_acc:.3f}")

import numpy as np
import pandas as pd

def get_feature_names(preprocessor, cat_features, num_features):
    cat_names = []
    if "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
    return cat_names + list(num_features)

feature_names = get_feature_names(preprocessor, categorical_features, numeric_features)


def auc_from_scores(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = (y == 1); neg = (y == 0)
    n_pos = pos.sum(); n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5 

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    sorted_s = s[order]
    diffs = np.r_[True, sorted_s[1:] != sorted_s[:-1], True]
    idx = np.flatnonzero(diffs)
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        if b - a > 1:
            avg = (a + 1 + b) / 2.0
            ranks[order[a:b]] = avg

    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

aucs = []
for j in range(X_test_np.shape[1]):
    col_auc = auc_from_scores(y_test, X_test_np[:, j])
    aucs.append(col_auc)

feat_auc_df = pd.DataFrame({
    "feature": feature_names,
    "auc": aucs
})
feat_auc_df["importance"] = np.abs(feat_auc_df["auc"] - 0.5)  
feat_auc_df = feat_auc_df.sort_values("importance", ascending=False).reset_index(drop=True)

def aggregate_ohe_auc(df_feat_auc: pd.DataFrame, categorical_features, numeric_features):
    rows, used = [], set()
    for col in categorical_features:
        prefix = f"{col}_"
        mask = df_feat_auc["feature"].str.startswith(prefix)
        if mask.any():
            rows.append({
                "feature": col,
                "importance": df_feat_auc.loc[mask, "importance"].sum()
            })
            used.update(df_feat_auc.loc[mask, "feature"].tolist())
    for col in numeric_features:
        if col in df_feat_auc["feature"].values:
            val = float(df_feat_auc.loc[df_feat_auc["feature"] == col, "importance"].values[0])
            rows.append({"feature": col, "importance": val})
    leftovers = df_feat_auc[~df_feat_auc["feature"].isin(used | set(numeric_features))]
    if not leftovers.empty:
        rows += leftovers[["feature","importance"]].to_dict("records")

    agg = (pd.DataFrame(rows)
             .groupby("feature", as_index=False)["importance"].sum()
             .sort_values("importance", ascending=False)
             .reset_index(drop=True))
    return agg

feat_auc_agg = aggregate_ohe_auc(feat_auc_df, categorical_features, numeric_features)

TOPK = 20
print("\n=== Univariate AUC — per encoded feature (Top {}): ===".format(TOPK))
print(feat_auc_df.head(TOPK)[["feature","auc","importance"]].to_string(index=False))
print("\n=== Univariate AUC — aggregated by original column (Top {}): ===".format(TOPK))
print(feat_auc_agg.head(TOPK).to_string(index=False))

from pathlib import Path
ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
feat_auc_df.to_csv(ART / "univariate_auc_per_dummy.csv", index=False)
feat_auc_agg.to_csv(ART / "univariate_auc_aggregated.csv", index=False)
print("\nSaved:",
      ART / "univariate_auc_per_dummy.csv",
      ART / "univariate_auc_aggregated.csv")

Path("artifacts").mkdir(exist_ok=True)
MODEL_PATH = "artifacts/aba_success_model.keras"
PIPE_PATH = "artifacts/preprocessor.joblib"
INTV_PATH = "artifacts/interventions.txt"

model.save(MODEL_PATH)
joblib.dump(preprocessor, PIPE_PATH)
with open(INTV_PATH, "w") as f:
    for itv in ALL_INTERVENTIONS:
        f.write(itv + "\n")

print("Saved:", MODEL_PATH, PIPE_PATH, INTV_PATH)

def load_artifacts():
    mdl = keras.models.load_model(MODEL_PATH)
    pp = joblib.load(PIPE_PATH)
    with open(INTV_PATH, "r") as f:
        intvs = [line.strip() for line in f if line.strip()]
    return mdl, pp, intvs

def recommend(context: dict, top_k: int = 3):
    """
    context: dict with keys matching the feature columns EXCEPT 'intervention'.
             e.g. {
               "gender": "Male",
               "diagnosis_level": "Level_2",
               "setting": "Home",
               "antecedent": "Task_demand",
               "behavior_type": "Non_compliance",
               "reinforcement": "Praise",
               "parent_training_level": "Medium",
               "medication_use": "No",
               "school_support": "Yes",
               "environment_noise_level": "Low",
               "reward_preference": "Edible",
               "age": 8,
               "behavior_intensity": 2,
               "behavior_duration_min": 6,
               "sibling_count": 1,
               "therapy_hours_week": 5,
               "behavior_frequency_last_week": 7,
               "parent_stress_level": 4,
               "session_hour": 16
             }
    """
    mdl, pp, intvs = load_artifacts()

    rows = []
    for itv in intvs:
        row = context.copy()
        row["intervention"] = itv
        for c in categorical_features:
            row.setdefault(c, "Unknown")
        for c in numeric_features:
            row.setdefault(c, float(np.nan))
        rows.append(row)

    cand_df = pd.DataFrame(rows)

    for c in numeric_features:
        if cand_df[c].isna().any():
            median = X_train[c].median() if c in X_train.columns else 0.0
            cand_df[c] = cand_df[c].fillna(median)

    X_cand = pp.transform(cand_df)
    preds = mdl.predict(X_cand, verbose=0).ravel() 
    cand_df["pred_success"] = preds
    cand_df = cand_df.sort_values("pred_success", ascending=False)

    top = cand_df[["intervention", "pred_success"]].head(top_k)
    return top.reset_index(drop=True)

sample = X_test.sample(1, random_state=7).iloc[0].to_dict()
if "intervention" in sample:
    sample.pop("intervention")

print("\nDEMO CONTEXT (no day_of_week):")
for k, v in sample.items():
    print(f"  {k}: {v}")

top3 = recommend(sample, top_k=3)
print("\nTOP-3 RECOMMENDED INTERVENTIONS (highest predicted success):")
print(top3.to_string(index=False))