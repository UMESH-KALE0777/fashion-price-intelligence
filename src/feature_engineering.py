import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CLEAN_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)

def prepare_features():
    logger.info("🎛️ Preparing features...")

    df = pd.read_csv(CLEAN_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    cat_cols = ["category", "brand"]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Train: (x={X_train.shape[0]}, f={X_train.shape[1]}), Test: (x={X_test.shape[0]}, f={X_test.shape[1]})")
    logger.info(f"Categorical columns: {cat_cols}")

    return X_train, X_test, y_train, y_test, cat_cols
