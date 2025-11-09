import pandas as pd

def add_price_position_and_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # price position vs brand/category means
    if "brand" in out.columns:
        out["brand_avg"] = out.groupby("brand")["predicted_price"].transform("mean")
    else:
        out["brand_avg"] = out["predicted_price"].mean()

    if "category" in out.columns:
        out["category_avg"] = out.groupby("category")["predicted_price"].transform("mean")
    else:
        out["category_avg"] = out["predicted_price"].mean()

    out["price_pos_brand"] = out["predicted_price"] / out["brand_avg"].replace(0, 1)
    out["price_pos_category"] = out["predicted_price"] / out["category_avg"].replace(0, 1)

    # simple tiering by percentile of predicted_price
    q = out["predicted_price"].quantile
    def tier(p):
        if p >= q(0.8): return "premium"
        if p >= q(0.5): return "mid"
        return "budget"
    out["tier"] = out["predicted_price"].apply(tier)
    return out

def top_n_by_tier(df: pd.DataFrame, n: int = 4, tier: str = "premium") -> pd.DataFrame:
    if "tier" not in df.columns:
        df = add_price_position_and_tier(df)
    return df[df["tier"] == tier].sort_values("predicted_price", ascending=False).head(n)
