"""Train several candidate models, track every run in MLflow, and register the best one.

Usage (from the repo root):
    python -m src.train

By default runs are stored in a local SQLite file (mlflow.db). Point MLFLOW_TRACKING_URI
at an MLflow server (e.g. http://localhost:5000) to log there instead.
"""
from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.features import (
    RANDOM_STATE,
    TARGET,
    SegmentRateBaseline,
    basic_clean,
    load_raw,
    make_pipeline,
    temporal_split,
)

EXPERIMENT_NAME = "hotel-cancellation"
REGISTERED_MODEL_NAME = "hotel-cancellation-model"
CHAMPION_ALIAS = "champion"
# MLflow saves sklearn models with skops (safer than pickle); our own transformer must be whitelisted
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype", "src.features.FeatureEngineer", "sklearn.tree._tree.Tree",
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
]


def candidate_models() -> dict:
    """name -> (model, preprocessing kind, hyper-parameters to log)."""
    xgb_base = dict(tree_method="hist", random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    return {
        "logistic_regression": (
            LogisticRegression(C=1.0, max_iter=2000), "onehot", {"C": 1.0}),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                   random_state=RANDOM_STATE),
            "ordinal", {"n_estimators": 200, "min_samples_leaf": 5}),
        "xgboost_shallow": (
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, **xgb_base),
            "ordinal", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1}),
        "xgboost_deep": (
            XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, **xgb_base),
            "ordinal", {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
                        "subsample": 0.8, "colsample_bytree": 0.8}),
    }


def evaluate(y_true, proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1_at_0.5": f1_score(y_true, (proba >= 0.5).astype(int)),
    }


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = basic_clean(load_raw())
    train, test = temporal_split(df)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    print(f"train={len(train):,} rows, test={len(test):,} rows (out-of-time split)")

    # 1) business-rule baseline, so every model can be compared against "what we do today"
    with mlflow.start_run(run_name="baseline_segment_rate"):
        baseline = SegmentRateBaseline().fit(X_train, y_train)
        metrics = evaluate(y_test, baseline.predict_proba(X_test)[:, 1])
        mlflow.log_param("model_type", "segment_rate_baseline")
        mlflow.log_metrics(metrics)
        print(f"{'baseline':<22} ROC-AUC={metrics['roc_auc']:.4f}")

    # 2) candidate ML models
    best = None
    for name, (model, kind, params) in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            pipe = make_pipeline(model, kind=kind).fit(X_train, y_train)
            metrics = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])
            mlflow.log_param("model_type", name)
            mlflow.log_param("preprocessing", kind)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            example = X_train.head(5)
            mlflow.sklearn.log_model(
                pipe, name="model", input_example=example,
                signature=infer_signature(example, pipe.predict_proba(example)[:, 1]),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            print(f"{name:<22} ROC-AUC={metrics['roc_auc']:.4f}")
            if best is None or metrics["roc_auc"] > best[1]:
                best = (run.info.run_id, metrics["roc_auc"], name)

    # 3) register the best run in the Model Registry and point the 'champion' alias at it
    run_id, auc, name = best
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    print(f"Registered {name} (ROC-AUC={auc:.4f}) as {REGISTERED_MODEL_NAME} "
          f"v{version.version} @{CHAMPION_ALIAS}")


if __name__ == "__main__":
    main()
