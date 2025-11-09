import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logger import get_logger
from src.config import CLEAN_DATA_PATH, MODEL_PATH, METADATA_PATH

logger = get_logger(__name__)

def train_model():
    logger.info("🚀 Training model (CatBoost)...")

    df = pd.read_csv(CLEAN_DATA_PATH)

    # Ensure all text columns remain text
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    # Separate target
    y = df["price"]
    X = df.drop(columns=["price"])

    # Detect categorical columns automatically
    cat_cols = list(X.select_dtypes(include=["object"]).columns)

    # Log-transform target for stable training
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Detected categorical columns: {cat_cols}")

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=900,
        loss_function="RMSE",
        verbose=50
    )

    # Train with auto categorical handling
    model.fit(X_train, y_train, cat_features=cat_cols)

    # Predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    r2 = r2_score(y_test, y_pred_log)
    mae = mean_absolute_error(np.expm1(y_test), y_pred)
    rmse = mean_squared_error(np.expm1(y_test), y_pred) ** 0.5

    logger.info(f"📊 R² Score: {r2:.4f}")
    logger.info(f"RMSE      : {rmse:.2f}")
    logger.info(f"MAE       : {mae:.2f}")

    # Save model & metadata
        # Save model & metadata
    model.save_model(MODEL_PATH)

    metadata = {
        "cat_cols": cat_cols,
        "feature_columns": list(X.columns)  # SAVE THIS
    }
    joblib.dump(metadata, METADATA_PATH)

    logger.info(f"✅ Model saved at {MODEL_PATH}")
    logger.info(f"✅ Metadata saved at {METADATA_PATH}")

    return model, {"r2": r2, "rmse": rmse, "mae": mae}
