"""Explains one prediction: which features pushed the cancellation probability up or down.

Impacts are contributions to the log-odds of cancellation: positive = pushes towards "will cancel",
negative = pushes towards "will stay". For XGBoost they are exact SHAP values (TreeSHAP, built into
XGBoost); for linear models they are coefficient x feature value.
"""
import math

import numpy as np
import pandas as pd

TOP_K = 5


def feature_contributions(pipeline, frame: pd.DataFrame) -> dict[str, float]:
    """Contribution of every model input column (after encoding) for the first row of `frame`."""
    preprocess, model = pipeline[:-1], pipeline[-1]
    X = np.asarray(preprocess.transform(frame), dtype=float)
    names = list(pipeline[-2].get_feature_names_out())
    if hasattr(model, "get_booster"):  # XGBoost
        import xgboost

        booster = model.get_booster()
        matrix = xgboost.DMatrix(X, feature_names=booster.feature_names)
        values = booster.predict(matrix, pred_contribs=True)[0][:-1]  # last column is the base value
    elif hasattr(model, "coef_"):  # linear model, e.g. logistic regression
        values = model.coef_[0] * X[0]
    else:
        raise TypeError(f"Cannot explain a {type(model).__name__} model")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def original_column(name: str, columns) -> str:
    """Map an encoded column back to the feature it came from: 'hotel_City Hotel' -> 'hotel'."""
    if name in columns:
        return name
    matches = [c for c in columns if name.startswith(f"{c}_")]
    return max(matches, key=len) if matches else name


def readable(value):
    """Turn numpy / pandas values into plain JSON values (NaN -> None)."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def top_factors(pipeline, frame: pd.DataFrame, k: int = TOP_K) -> list[dict]:
    """The k features with the largest impact on this prediction, largest first."""
    features = pipeline[0].transform(frame).iloc[0]  # values after feature engineering, before encoding
    totals: dict[str, float] = {}
    for name, impact in feature_contributions(pipeline, frame).items():
        column = original_column(name, features.index)
        totals[column] = totals.get(column, 0.0) + impact
    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": column, "value": readable(features.get(column)), "impact": round(impact, 4)}
        for column, impact in ranked[:k]
        if abs(impact) > 1e-6
    ]
