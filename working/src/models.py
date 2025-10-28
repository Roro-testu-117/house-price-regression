# ============================================
# models.py
# モデル学習に関する関数群
# ============================================


# ---------- ライブラリのインポート ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error as rmse
from typing import Tuple, Dict, Any
import xgboost as xgb


# =====================================================
# XGBoost
# =====================================================
def xgb_train_cv(
    input_x: pd.DataFrame,
    input_y: pd.Series,
    input_test_x: pd.DataFrame,
    params: Dict[str, Any],
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    XGBoostを用いてクロスバリデーションを行い、モデルの性能を評価・予測する関数。

    Example(タプルアンパック):
    imp, scores, test_preds, val_preds = xgb_train_cv(
      X_train, y_train, X_test, params, n_splits=5
    )
    """

    imp = pd.DataFrame()  # 特徴量の重要度を格納するdf
    scores = []  # 評価指標のスコアを格納するリスト
    test_preds = np.zeros(len(input_test_x))  # テストデータの予測値を格納するリスト
    val_preds = np.zeros(len(input_x))  # 予測値を格納するリスト

    # データを学習用と評価用に分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv = list(kf.split(input_x, input_y))

    for nfold in range(n_splits):
        tr_idx, val_idx = cv[nfold]
        x_tr, x_val = input_x.iloc[tr_idx], input_x.iloc[val_idx]
        y_tr, y_val = input_y.iloc[tr_idx], input_y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        print("学習中...")
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_tr, y_tr), (x_val, y_val)],
            verbose=100,
        )

        print("予測 & 精度計算中...")
        y_tr_preds = model.predict(x_tr)
        y_val_preds = model.predict(x_val)
        metric_tr = round(rmse(y_tr, y_tr_preds), 5)
        metric_val = round(rmse(y_val, y_val_preds), 5)
        print("[RSME] tr: {:.5f}, val: {:.5f}".format(metric_tr, metric_val))
        scores.append([nfold, metric_tr, metric_val])  # 結果を格納
        # 検証データの予測値を該当のindexの場所に格納
        val_preds[val_idx] = y_val_preds

        # testデータに対する予測値
        y_test_preds = model.predict(input_test_x)
        test_preds += y_test_preds

        # 特徴量重要度
        _imp = pd.DataFrame(
            {"col": input_x.columns, "imp": model.feature_importances_, "nfold": nfold}
        )
        # print(_imp)
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    # 検証データ
    scores = np.array(scores)

    # foldごとの予測値の平均
    test_preds /= n_splits

    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp_mean", "imp_std"]
    imp = imp.reset_index(drop=False)

    return imp, scores, test_preds, val_preds
