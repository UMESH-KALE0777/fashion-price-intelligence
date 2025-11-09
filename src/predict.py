import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from pathlib import Path
from src.config import CLEAN_DATA_PATH, MODEL_PATH, METADATA_PATH
from src.logger import get_logger

logger = get_logger(__name__)

def predict_new_data():
    logger.info("🔮 Starting prediction...")

    # Load the trained model
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # Load metadata
    meta = joblib.load(METADATA_PATH)
    cat_cols = meta["cat_cols"]
    feature_columns = meta["feature_columns"]

    # Load clean dataset
    df = pd.read_csv(CLEAN_DATA_PATH)

    # Ensure all required columns are present
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    # Select model input features
    df_model = df[feature_columns].copy()

    # Predict (model outputs log(price))
    preds_log = model.predict(df_model)

    # Convert log → actual price
    df["predicted_price"] = np.expm1(preds_log)

    # Save predictions
    predictions_path = Path(CLEAN_DATA_PATH).parent / "predictions.csv"
    df.to_csv(predictions_path, index=False)
    logger.info(f"✅ Predictions saved at {predictions_path}")

    # Best Product (Highest predicted sale price)
    best = df.loc[df["predicted_price"].idxmax()]
    logger.info(f"🏆 Best Product Recommendation:\n{best}")

    return df, best
