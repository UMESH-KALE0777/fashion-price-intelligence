# ----------------------------
# Fashion Price Intelligence - Luxury Black & Gold Edition
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


# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Fashion Price Intelligence",
    page_icon="👑",
    layout="wide"
)

# ---------- THEME COLORS (Black & Gold) ----------
PRIMARY = "#D4AF37"  # Gold
SECONDARY = "#1A1A1A"  # Rich Black
BG = "#000000"  # Full black background

# ---------- PREMIUM UI ----------
st.markdown(f"""
<style>
body {{
    background-color: {BG};
}}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1400px;
}}
.header-box {{
    background: linear-gradient(135deg, {SECONDARY}, {PRIMARY});
    padding: 35px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    margin-bottom: 35px;
}}
.header-box h1 {{
    margin: 0;
    font-weight: 700;
    font-size: 42px;
}}
.header-box p {{
    margin-top: 6px;
    font-size: 16px;
    opacity: .9;
}}

.metric-card {{
    background: #111;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #333;
    text-align: center;
}}
.metric-card h3 {{
    color: #aaa;
    font-size: 14px;
    margin-bottom: 6px;
    font-weight: 500;
}}
.metric-card .value {{
    font-size: 28px;
    color: {PRIMARY};
    font-weight: 800;
}}

.stButton>button {{
    background-color: {PRIMARY};
    color: black;
    padding: 0.6rem 1.3rem;
    border-radius: 10px;
    border: none;
    font-weight: 650;
    transition: .2s;
}}
.stButton>button:hover {{
    background-color: white;
    transform: scale(1.04);
}}

.dataframe {{
    background: white !important;
}}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="header-box">
    <h1>👑 Fashion Price Intelligence</h1>
    <p>Upload → Train → Predict → Market Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
BASE = Path(__file__).resolve().parent.parent
SAMPLE = BASE / "data" / "raw_data.csv"

st.sidebar.header("📦 Upload Data")
file = st.sidebar.file_uploader("Upload Fashion Store CSV", type=["csv"])
use_sample = st.sidebar.toggle("Use sample dataset", value=not bool(file))

if file:
    df = pd.read_csv(file)
elif use_sample and SAMPLE.exists():
    df = pd.read_csv(SAMPLE)
else:
    st.stop()

df = df.apply(lambda col: col.astype(str).str.strip() if col.dtype == "object" else col)

cols = list(df.columns)


# ---------- COLUMN MAPPING ----------
st.subheader("Column Mapping")
c1, c2, c3, c4 = st.columns(4)

col_pid = c1.selectbox("Product ID (optional)", ["-- none --"] + cols)
col_cat = c2.selectbox("Category", cols)
col_brand = c3.selectbox("Brand", cols)
col_price = c4.selectbox("Target Price", cols)

col_rating = st.selectbox("Rating (optional)", ["-- none --"] + cols)

include_pid = (col_pid != "-- none --") and st.checkbox("Use Product ID as Feature", value=True)


# ---------- FEATURES ----------
def build_features(data):
    df2 = data.copy()

    df2[col_price] = pd.to_numeric(df2[col_price], errors="coerce")
    df2 = df2.dropna(subset=[col_price])

    numeric_cols = []
    if col_rating != "-- none --":
        df2[col_rating] = pd.to_numeric(df2[col_rating], errors="coerce")
        numeric_cols.append(col_rating)

    df2["category_avg_price"] = df2.groupby(col_cat)[col_price].transform("mean")
    df2["brand_avg_price"] = df2.groupby(col_brand)[col_price].transform("mean")
    df2["category_popularity"] = df2.groupby(col_cat)[col_price].transform("count")
    df2["brand_popularity"] = df2.groupby(col_brand)[col_price].transform("count")
    df2["brand_category_combo"] = df2[col_brand].astype(str) + "_" + df2[col_cat].astype(str)

    feat = []
    if include_pid:
        feat.append(col_pid)
    feat += [col_cat, col_brand, "brand_category_combo"] + numeric_cols
    feat += ["category_avg_price", "brand_avg_price", "category_popularity", "brand_popularity"]

    X = df2[feat].copy()
    y = df2[col_price]

    # Convert cat features to string (important fix)
    for c in [col_cat, col_brand, "brand_category_combo"] + ([col_pid] if include_pid else []):
        if c in X:
            X[c] = X[c].astype(str)

    cat_idx = [X.columns.get_loc(c) for c in X.columns if X[c].dtype == object]
    return X, y, feat, cat_idx


# ---------- TRAIN ----------
X, y, feat, cat_idx = build_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = CatBoostRegressor(iterations=800, depth=8, learning_rate=0.08, loss_function="RMSE", verbose=False)
model.fit(X_train, y_train, cat_features=cat_idx)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)


# ---------- METRICS UI ----------
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='metric-card'><h3>TRAIN ROWS</h3><div class='value'>{len(X_train)}</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><h3>TEST ROWS</h3><div class='value'>{len(X_test)}</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><h3>FEATURE COUNT</h3><div class='value'>{len(feat)}</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card'><h3>R² SCORE</h3><div class='value'>{r2:.3f}</div></div>", unsafe_allow_html=True)

st.markdown("### Actual vs Predicted")
chart_df = pd.DataFrame({"Actual": y_test, "Predicted": pred})
st.plotly_chart(px.scatter(chart_df, x="Actual", y="Predicted", color_discrete_sequence=[PRIMARY]), use_container_width=True)


# ---------- FULL PREDICTION ----------
X_full, _, _, _ = build_features(df)
df["predicted_price"] = model.predict(X_full)
st.markdown("### Predictions Table")
st.dataframe(df.head(25), use_container_width=True)


# ---------- DOWNLOAD ----------
st.download_button("Download Prediction CSV", df.to_csv(index=False), "predictions.csv")


# ---------- PREMIUM BESTSELLER INSIGHTS ----------
st.markdown("## 👑 Bestseller Strategy Insights (Premium)")

insight_df = df.copy()

# 1) Category Level Performance
cat_summary = insight_df.groupby(col_cat).agg(
    avg_price=("predicted_price", "mean"),
    avg_rating=(col_rating, "mean") if col_rating != "-- none --" else ("predicted_price", "count"),
    popularity=("predicted_price", "count")
).reset_index()

st.markdown("### Category Performance Overview")
st.dataframe(cat_summary, use_container_width=True)

# Category insights chart
fig_cat = px.bar(
    cat_summary,
    x=col_cat,
    y="popularity",
    color="avg_price",
    title="Category Popularity vs Average Predicted Price",
    color_continuous_scale="tealrose",   # premium scale
    height=420
)
fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_cat, use_container_width=True)


# 2) Brand Strength Index
brand_summary = insight_df.groupby(col_brand).agg(
    avg_price=("predicted_price", "mean"),
    total_items=("predicted_price", "count"),
    popularity=("predicted_price", "count"),
).reset_index()

brand_summary["brand_strength_index"] = (
    brand_summary["avg_price"] * 0.6 + brand_summary["popularity"] * 0.4
)

st.markdown("### Brand Strength & Price Influence")
st.dataframe(brand_summary.sort_values("brand_strength_index", ascending=False).head(15), use_container_width=True)

# Brand strength chart
fig_brand = px.scatter(
    brand_summary,
    x="popularity",
    y="avg_price",
    size="brand_strength_index",
    hover_name=col_brand,
    title="Brand Strength Map (Premium Analysis)",
    color="brand_strength_index",
    color_continuous_scale="tealrose",   # premium scale
    height=420
)
fig_brand.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_brand, use_container_width=True)



# 3) Bestseller Identification
bestsellers = insight_df.sort_values("predicted_price", ascending=False).head(10)

st.markdown("### 🔥 Top Predicted Revenue Products")
st.dataframe(bestsellers[[col_pid, col_cat, col_brand, col_price, "predicted_price"]] if col_pid != "-- none --"
             else bestsellers[[col_cat, col_brand, col_price, "predicted_price"]],
             use_container_width=True)



# 4) Pricing Strategy Recommendation
st.markdown("### 💰 Price Strategy Recommendation Engine")

avg_predicted = df["predicted_price"].mean()
current_price = df[col_price].mean()

if current_price < avg_predicted * 0.85:
    st.success("Recommendation: **Increase your pricing slightly. Your brand is undervalued.**")
elif current_price > avg_predicted * 1.20:
    st.warning("Recommendation: **Consider offering discounts. Your pricing is above competitive market predictions.**")
else:
    st.info("Recommendation: **Your pricing is aligned with the predicted market range. Maintain strategic stability.**")



