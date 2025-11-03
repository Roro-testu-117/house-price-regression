# ============================================
# optuna.py
# Optunaによるハイパーパラメータチューニング
# ============================================

import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error as rmse
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# =====================================================
# LightGBM Optuna Optimization
# =====================================================
def optimize_lgb(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    params_base: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    LightGBMのハイパーパラメータをOptunaで最適化するための目的関数。
    """

    # パラメータの探索範囲
    params_tuning = {
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.6, 0.95, step=0.05
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.6, 0.95, step=0.05
        ),
        "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split", 1e-8, 1.0),
        # 余裕があればlambda_l1, lambda_l2も調整する
        # "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1.0),
        # "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-6, 10.0),
    }
    params_tuning.update(params_base)

    # モデル学習、評価
    metrics = []  # 評価指標を格納するためのリスト

    # データを学習用と評価用に分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = list(kf.split(X, y))

    # 交差検証法
    for nfold in range(n_splits):
        tr_idx, val_idx = cv[nfold][0], cv[nfold][1]
        x_tr, x_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # モデルの学習
        model = lgb.LGBMRegressor(**params_tuning)

        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_tr, y_tr), (x_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(100),
            ],
        )

        # モデルの予測
        y_val_pred = model.predict(x_val)

        # 正解率の算出
        val_score = rmse(y_val, y_val_pred)
        metrics.append(val_score)

    # 平均スコアを返す
    return np.mean(metrics)


# =====================================================
# XGBoost Optuna Optimization
# =====================================================
def optimize_xgb(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    params_base: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    XGBoostのハイパーパラメータをOptunaで最適化するための目的関数。
    """

    params_tuning = {
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95, step=0.05),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 0.95, step=0.05
        ),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        # 余裕があればalpha, lambdaも調整する
        # 'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        # 'lambda': trial.suggest_loguniform('lambda', 1e-6, 10.0),
    }
    params_tuning.update(params_base)

    # モデル学習、評価
    metrics = []  # 評価指標を格納するためのリスト

    # データを学習用と評価用に分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = list(kf.split(X, y))

    # 交差検証法
    for nfold in range(n_splits):
        tr_idx, val_idx = cv[nfold][0], cv[nfold][1]
        x_tr, x_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # XGBoost モデルのインスタンス化
        model = xgb.XGBRegressor(**params_tuning)

        # 学習
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_tr, y_tr), (x_val, y_val)],
            verbose=100,
        )

        # モデルの予測
        y_val_pred = model.predict(x_val)

        # 評価指標の算出
        val_score = metric_function(y_val, y_val_pred)
        metrics.append(val_score)

    # 平均スコアを返す
    return np.mean(metrics)
