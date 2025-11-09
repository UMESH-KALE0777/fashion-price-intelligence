# ----------------------------
# Fashion Price Intelligence - Streamlit Dashboard (Final Stable Build)
# ----------------------------
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor


# ---------- PAGE UI ----------
st.set_page_config(
    page_title="Fashion Price Intelligence",
    page_icon="👜",
    layout="wide",
)

PRIMARY = "#0e7490"
ACCENT  = "#0891b2"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
  .metric-card {{
      background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
      color: white; padding: 20px; border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15); 
  }}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header">
  <h1 style="margin:0;">Fashion Price Intelligence</h1>
  <div>Upload → Map Columns → Train → Predict</div>
</div>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "raw_data.csv"

st.sidebar.header("📦 Upload Data")
uploaded = st.sidebar.file_uploader("Upload your fashion store CSV", type=["csv"])
use_sample = st.sidebar.toggle("Use sample dataset", value=not bool(uploaded))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
else:
    st.stop()

df = df.copy()
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.strip()

cols = list(df.columns)


# ---------- COLUMN MAPPING ----------
st.subheader("Column Mapping")

c1, c2, c3, c4 = st.columns(4)
col_pid   = c1.selectbox("Product ID (optional)", ["-- none --"] + cols)
col_cat   = c2.selectbox("Category", cols)
col_brand = c3.selectbox("Brand", cols)
col_price = c4.selectbox("Target Price", cols)

col_rating = st.selectbox("Rating (optional)", ["-- none --"] + cols)
include_pid_as_feature = (col_pid != "-- none --") and st.checkbox("Use Product ID as feature", False)

# **FIX: Prevent same column selection**
if len({col_cat, col_brand, col_price}) < 3:
    st.error("Category, Brand, and Target Price must be different columns.")
    st.stop()


# ---------- FEATURE ENGINEERING ----------
def build_features(data):
    df2 = data.copy()
    df2[col_price] = pd.to_numeric(df2[col_price], errors="coerce")
    df2 = df2.dropna(subset=[col_price])

    numeric_cols = []
    if col_rating != "-- none --":
        df2[col_rating] = pd.to_numeric(df2[col_rating], errors="coerce")
        numeric_cols.append(col_rating)

    df2["category_avg_price"]  = df2.groupby(col_cat)[col_price].transform("mean")
    df2["brand_avg_price"]     = df2.groupby(col_brand)[col_price].transform("mean")
    df2["category_popularity"] = df2.groupby(col_cat)[col_price].transform("count")
    df2["brand_popularity"]    = df2.groupby(col_brand)[col_price].transform("count")
    df2["brand_category_combo"] = df2[col_brand].astype(str) + "_" + df2[col_cat].astype(str)

    features = []
    if include_pid_as_feature and col_pid in df2.columns:
        features.append(col_pid)

    features += [col_cat, col_brand, "brand_category_combo"]
    features += numeric_cols
    features += ["category_avg_price", "brand_avg_price", "category_popularity", "brand_popularity"]

    X = df2[features].copy()
    y = df2[col_price]

    # **FIX: Make sure categorical columns are actually treated as strings**
    for c in [col_pid if include_pid_as_feature else None, col_cat, col_brand, "brand_category_combo"]:
        if c in X:
            X[c] = X[c].astype(str)

    # **FIX: Safe categorical detection (works in Streamlit Cloud)**
    cat_idx = [X.columns.get_loc(c) for c in X.columns if str(X[c].dtype) == "object"]

    return X, y, features, cat_idx


# ---------- TRAIN ----------
@st.cache_data
def train_model(df):
    X, y, features, cat_idx = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=700, learning_rate=0.08, depth=8, loss_function="RMSE", verbose=False)
    model.fit(X_train, y_train, cat_features=cat_idx)

    pred = model.predict(X_test)
    return model, features, X_test, y_test, pred

model, features, X_test, y_test, pred = train_model(df)


# ---------- METRICS ----------
st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Train Rows", len(df)*0.8)
m2.metric("Test Rows", len(df)*0.2)
m3.metric("R² Score", f"{r2_score(y_test, pred):.3f}")
m4.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.1f}")


# ---------- PREDICTIONS TABLE ----------
st.subheader("Predictions")
X_full, _, _, _ = build_features(df)
df["predicted_price"] = model.predict(X_full)
st.dataframe(df.head(20), use_container_width=True)
st.download_button("Download predictions CSV", df.to_csv(index=False), "predictions.csv")


# ---------- SINGLE PREDICT ----------
st.subheader("Predict Single Product")
category = st.selectbox("Category", sorted(df[col_cat].unique()))
brand = st.selectbox("Brand", sorted(df[col_brand].unique()))
rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)

one = {
    col_cat: category,
    col_brand: brand,
    "brand_category_combo": f"{brand}_{category}",
    "category_avg_price": df.loc[df[col_cat] == category, col_price].mean(),
    "brand_avg_price": df.loc[df[col_brand] == brand, col_price].mean(),
    "category_popularity": (df[col_cat] == category).sum(),
    "brand_popularity": (df[col_brand] == brand).sum(),
}

tmp = pd.DataFrame([one]).reindex(columns=features, fill_value=0)
price_pred = float(model.predict(tmp)[0])
st.success(f"💰 Predicted Price: ₹ {price_pred:,.0f}")
