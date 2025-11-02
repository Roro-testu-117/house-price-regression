# ============================================
# optuna.py
# Optunaによるハイパーパラメータチューニング
# ============================================

import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Dict, Any
import numpy as np
import pandas as pd


# =====================================================
# LightGBM Optuna Optimization
# =====================================================
def optimize_lgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    LightGBMのハイパーパラメータをOptunaで最適化するための目的関数。
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        verbose_eval=False,
        num_boost_round=1000,
        early_stopping_rounds=50,
    )

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse


# =====================================================
# XGBoost Optuna Optimization
# =====================================================
def optimize_xgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    XGBoostのハイパーパラメータをOptunaで最適化するための目的関数。
    """
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-3, 10.0),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=False,
    )

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse
