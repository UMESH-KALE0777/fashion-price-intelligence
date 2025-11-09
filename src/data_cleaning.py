import pandas as pd
from src.logger import get_logger
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH

logger = get_logger(__name__)

def clean_data():
    logger.info("🧹 Starting data cleaning process...")

    df = pd.read_csv(RAW_DATA_PATH)

    # Standardizing column names
    df.columns = df.columns.str.lower().str.strip()

    # Fill missing values
    df["category"] = df["category"].fillna("Unknown")
    df["brand"] = df["brand"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Remove out-of-range prices
    df = df[(df["price"] > 50) & (df["price"] < 20000)]

    # **New Feature Engineering**

    # 1. Category Average Price
    df["category_avg_price"] = df.groupby("category")["price"].transform("mean")

    # 2. Brand Average Price
    df["brand_avg_price"] = df.groupby("brand")["price"].transform("mean")

    # 3. Category Popularity
    df["category_popularity"] = df.groupby("category")["price"].transform("count")

    # 4. Brand Popularity
    df["brand_popularity"] = df.groupby("brand")["price"].transform("count")

    # 5. Interaction Feature
    df["brand_category_combo"] = df["brand"] + "_" + df["category"]

    df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info(f"✅ Cleaned data saved at {CLEAN_DATA_PATH}, shape: {df.shape}")
