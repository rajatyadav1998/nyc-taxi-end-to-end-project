import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="NYC Taxi — Fare & Payment Predictor", layout="wide")

MODEL_DIR = Path("models")
CLF_DIR = Path("models/clf_tuned")
ART_DIR = Path("/content")

st.title("NYC Taxi — Fare Prediction & Payment Type (Demo)")

# -------------- load models (prefer tuned, fallback to sample)
@st.cache_resource
def load_models():
    models = {}
    # regression
    try:
        models['reg'] = joblib.load(MODEL_DIR / "xgb_fare_model_tuned.pkl")
    except Exception:
        try:
            models['reg'] = joblib.load("/content/xgb_fare_model_sample.pkl")
        except Exception:
            models['reg'] = None
    # classification (GB)
    try:
        models['clf_gb'] = joblib.load(CLF_DIR / "GradientBoosting_tuned.pkl")
    except Exception:
        try:
            models['clf_gb'] = joblib.load("models/clf/Gradient_Boosting.pkl")
        except Exception:
            models['clf_gb'] = None
    # logistic (optional)
    try:
        models['clf_logit'] = joblib.load(CLF_DIR / "LogisticRegression_tuned.pkl")
    except Exception:
        models['clf_logit'] = None
    return models

models = load_models()

# -------------- sidebar inputs (basic features)
st.sidebar.header("Input features")
def ui_input():
    VendorID = st.sidebar.selectbox("VendorID", [1,2], index=0)
    passenger_count = st.sidebar.number_input("passenger_count", min_value=0, max_value=10, value=1)
    trip_distance = st.sidebar.number_input("trip_distance (miles)", min_value=0.0, value=3.0, format="%.2f")
    RatecodeID = st.sidebar.selectbox("RatecodeID", [1,2,3,4,5,6], index=0)
    PULocationID = st.sidebar.number_input("PULocationID", min_value=0, value=100)
    DOLocationID = st.sidebar.number_input("DOLocationID", min_value=0, value=200)
    trip_duration_min = st.sidebar.number_input("trip_duration_min", min_value=0.1, value=10.0, format="%.2f")
    hour = st.sidebar.slider("pickup hour", 0, 23, 9)
    day_of_week = st.sidebar.slider("day_of_week (0=Mon)", 0, 6, 2)
    speed_mph = st.sidebar.number_input("speed_mph", min_value=0.0, value=15.0, format="%.2f")
    is_long_trip = st.sidebar.checkbox("is_long_trip (distance>10mi)", value=(trip_distance>10.0))
    return {
        "VendorID": VendorID,
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "RatecodeID": RatecodeID,
        "PULocationID": PULocationID,
        "DOLocationID": DOLocationID,
        "trip_duration_min": trip_duration_min,
        "hour": hour,
        "day_of_week": day_of_week,
        "speed_mph": speed_mph,
        "is_long_trip": int(is_long_trip)
    }

inp = ui_input()
X_df = pd.DataFrame([inp])

st.subheader("Input preview")
st.dataframe(X_df.T, width=350)

# -------------- Predictions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Fare prediction (regression)")
    if models.get('reg') is not None:
        try:
            fare_pred = models['reg'].predict(X_df)[0]
            st.metric("Predicted fare (USD)", f"{fare_pred:.2f}")
        except Exception as e:
            st.error("Regression prediction failed: " + str(e))
    else:
        st.warning("Regression model not found.")

    # show regression SHAP plot if available
    shap_path = ART_DIR / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP summary (regression)", use_column_width=True)

with col2:
    st.subheader("Payment type (classification)")
    # show probabilities from GB and logistic (if present)
    if models.get('clf_gb') is not None:
        try:
            p_gb = models['clf_gb'].predict_proba(X_df)[:,1][0]
            st.metric("P(credit) - GradientBoosting", f"{p_gb:.3f}")
        except Exception as e:
            st.error("Classification (GB) failed: " + str(e))
    else:
        st.warning("GB classifier not found.")

    if models.get('clf_logit') is not None:
        try:
            p_log = models['clf_logit'].predict_proba(X_df)[:,1][0]
            st.metric("P(credit) - Logistic", f"{p_log:.3f}")
        except Exception as e:
            st.error("Classification (Logit) failed: " + str(e))

    # show classification SHAP plot if available
    shap_clf_path = ART_DIR / "shap_clf_summary.png"
    if shap_clf_path.exists():
        st.image(str(shap_clf_path), caption="SHAP summary (classification)", use_column_width=True)

# -------------- Metrics & dataset preview
st.markdown("---")
st.subheader("Dataset sample & metrics (from CSV)")
csv_path = "/content/nyc_gold_sample.csv"
if Path(csv_path).exists():
    df_sample = pd.read_csv(csv_path)
    st.write("Sample rows:", df_sample.shape[0])
    st.dataframe(df_sample.head(10))
else:
    st.info("Local sample CSV not found at /content/nyc_gold_sample.csv")

# -------------- Footer: S3 note
st.markdown("---")
st.caption("Models and artifacts loaded from the Colab workspace. If you want the app to load models directly from S3, replace joblib.load paths with boto3 download logic and point to s3://nyc-taxi-raw-rajat/models/ ...")


# ========================== START PREDICTION BLOCK ==========================

# ===========================
# SAFE PREDICTION BLOCK (AUTO-INJECTED)
# ===========================

st.header("🔮 Predictions")

TRAIN_FEATURES = [
    "VendorID","passenger_count","trip_distance","RatecodeID",
    "PULocationID","DOLocationID","trip_duration_min",
    "hour","day_of_week","speed_mph","is_long_trip"
]

def make_aligned_input(model, X_raw, fallback=TRAIN_FEATURES):
    expected = None
    try:
        expected = list(getattr(model, "feature_names_in_", []))
    except:
        expected = None
    if not expected:
        expected = fallback.copy()
    X = X_raw.copy()
    for c in expected:
        if c not in X.columns:
            X[c] = 0
    X = X[expected]
    return X

# --- build base dataframe from UI inputs ---
X_raw = pd.DataFrame([inp]).copy()

# ===== Regression =====
st.subheader("💵 Fare Prediction (Regression)")
reg = models.get("reg")
if reg:
    try:
        X_reg = make_aligned_input(reg, X_raw)
        pred = reg.predict(X_reg)[0]
        st.metric("Predicted Fare", f"${pred:.2f}")
    except Exception as e:
        st.error(f"Regression failed: {e}")
else:
    st.warning("Regression model not loaded.")

# ===== Classification =====
st.subheader("💳 Payment Type Prediction (Classification)")

clf = models.get("clf_gb")
if clf:
    try:
        X_cls = make_aligned_input(clf, X_raw)
        if "payment_type" in X_cls.columns:
            X_cls = X_cls.drop(columns=["payment_type"])
        prob = clf.predict_proba(X_cls)[0][1]
        label = "Credit Card" if prob >= 0.5 else "Cash"
        st.metric("Predicted Method", f"{label} ({prob:.3f})")
    except Exception as e:
        st.error(f"Classification failed: {e}")
else:
    st.warning("Classifier model not loaded.")

# ========================== END PREDICTION BLOCK ==========================
