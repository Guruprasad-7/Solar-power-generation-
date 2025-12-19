import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------
st.set_page_config(page_title="Solar Power Generation",
                   layout="wide",
                   initial_sidebar_state="expanded")

# -------------------------------------------------------
# Load Data + Model + Scaler
# -------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Convert datetime columns if present
    for col in df.columns:
        if any(t in col.lower() for t in ["date", "time", "datetime"]):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

# ---- FILE PATHS ----
DATA_PATH = r"C:\Users\Runku\DS Project 2\solarpowergeneration.csv"
MODEL_PATH = r"C:\Users\Runku\DS Project 2\gradient_boosting_model.joblib"
SCALER_PATH = r"C:\Users\Runku\DS Project 2\scaler.joblib"

# Load assets
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# -------------------------------------------------------
# Identify Features & Target
# -------------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target_col = numeric_cols[-1]          # last numeric column = target
feature_cols = numeric_cols[:-1]       # all others are features

# If model feature count doesn't match CSV → auto-correct
model_features = len(model.feature_importances_)
if len(feature_cols) != model_features:
    st.warning(f"⚠ Auto-adjusting feature list to match model ({model_features} features)")
    feature_cols = feature_cols[:model_features]

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.title("☀️ Solar Power Generation")
st.markdown("Interactive dashboard + ML prediction for solar power generation.")

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.header("Filters")

datetime_cols = [c for c in df.columns if "date" in c.lower()]
if datetime_cols:
    dt = datetime_cols[0]
    min_d, max_d = df[dt].min(), df[dt].max()
    date_range = st.sidebar.date_input("Date Range", [min_d, max_d])

    if len(date_range) == 2:
        start, end = date_range
        df = df[(df[dt] >= pd.to_datetime(start)) & (df[dt] <= pd.to_datetime(end))]

selected_features = st.sidebar.multiselect("Select Features for Scatter Matrix",
                                           feature_cols,
                                           default=feature_cols[:4])

# -------------------------------------------------------
# Top Stats
# -------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{len(df):,}")
col2.metric("Feature Count", f"{len(feature_cols)}")
col3.metric("Model Loaded", "YES" if model else "NO")

# -------------------------------------------------------
# Data Visualizations
# -------------------------------------------------------
st.header("📊 Data Visualizations")

# Time series chart
if datetime_cols:
    dt = datetime_cols[0]
    st.subheader("Time Series — Target Variable")
    fig = px.line(df, x=dt, y=target_col, title=f"{target_col} Over Time")
    st.plotly_chart(fig, use_container_width=True)

# Scatter matrix
if len(selected_features) >= 2:
    st.subheader("Scatter Matrix")
    fig = px.scatter_matrix(df[selected_features].sample(min(300, len(df))))
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Feature Importance (Auto-corrected)
# -------------------------------------------------------
st.header("🌟 Model Feature Importance")

importances = model.feature_importances_

fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
fig = px.bar(fi, title="Feature Importance")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Prediction Form
# -------------------------------------------------------
st.header("🔮 Make a Prediction")

with st.form("prediction_form"):
    inputs = {}

    for f in feature_cols:
        mean = float(df[f].mean())
        minv = float(df[f].min())
        maxv = float(df[f].max())

        inputs[f] = st.number_input(
            f, value=mean, min_value=minv, max_value=maxv,
            step=(maxv - minv) / 100
        )

    submit = st.form_submit_button("Predict")

if submit:
    X = pd.DataFrame([inputs])

    # Ensure proper ordering for model
    X = X[feature_cols]

    # Apply scaler (if exists)
    X_scaled = scaler.transform(X) if scaler else X

    pred = model.predict(X_scaled)[0]
    st.success(f"Predicted {target_col}: **{pred:.3f}**")

# -------------------------------------------------------
# Download Predictions
# -------------------------------------------------------
st.header("📥 Download Model Predictions for Entire Dataset")

if st.button("Generate Predictions"):
    X = df[feature_cols]
    X_scaled = scaler.transform(X) if scaler else X

    df["Prediction"] = model.predict(X_scaled)

    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV",
                       csv,
                       "solar_predictions.csv",
                       "text/csv")

