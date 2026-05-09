"""Юнит-тесты для CP2.

Покрывают предобработку, метрики и harness экспериментов на синтетике.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from src.experiments import (
    define_experiments,
    results_to_table,
    select_and_finalize,
)
from src.modeling import clf_metrics, find_best_threshold
from src.preprocessing import (
    RAW_FEATURES,
    SEED,
    TARGET_BIN,
    clip_outliers,
    drop_duplicates,
    encode_target,
    handle_missing,
    split_data,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 1500
    data: dict[str, np.ndarray | list[str]] = {col: rng.standard_normal(n) for col in RAW_FEATURES}
    data["company_name"] = [f"C_{i // 5}" for i in range(n)]
    data["year"] = rng.integers(1999, 2019, n)
    data["status_label"] = np.where(rng.random(n) < 0.07, "failed", "alive")
    return pd.DataFrame(data)


@pytest.fixture
def encoded_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return encode_target(sample_df)


@pytest.fixture
def clean_df(encoded_df: pd.DataFrame) -> pd.DataFrame:
    df = drop_duplicates(encoded_df)
    df = handle_missing(df)
    df = clip_outliers(df, RAW_FEATURES)
    return df


@pytest.fixture
def splits(clean_df: pd.DataFrame):
    X = clean_df[RAW_FEATURES]
    y = clean_df[TARGET_BIN]
    return split_data(X, y)


def test_encode_target_creates_binary_column(encoded_df: pd.DataFrame) -> None:
    assert TARGET_BIN in encoded_df.columns
    assert set(encoded_df[TARGET_BIN].unique()).issubset({0, 1})


def test_drop_duplicates_returns_dataframe(encoded_df: pd.DataFrame) -> None:
    result = drop_duplicates(encoded_df)
    assert isinstance(result, pd.DataFrame)
    assert result.duplicated().sum() == 0


def test_clip_outliers_respects_quantiles(encoded_df: pd.DataFrame) -> None:
    clipped = clip_outliers(encoded_df, RAW_FEATURES, lower_q=0.05, upper_q=0.95)
    for col in RAW_FEATURES:
        lo = encoded_df[col].quantile(0.05)
        hi = encoded_df[col].quantile(0.95)
        assert clipped[col].min() >= lo - 1e-9
        assert clipped[col].max() <= hi + 1e-9


def test_split_keeps_all_rows_and_stratifies(clean_df: pd.DataFrame) -> None:
    X = clean_df[RAW_FEATURES]
    y = clean_df[TARGET_BIN]
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
    assert len(X_tr) + len(X_v) + len(X_te) == len(X)
    rates = [y_tr.mean(), y_v.mean(), y_te.mean()]
    assert max(rates) - min(rates) < 0.05


def test_clf_metrics_returns_six_keys() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.85])
    m = clf_metrics(y_true, y_pred, y_prob, prefix="x_")
    assert set(m) == {"x_accuracy", "x_precision", "x_recall", "x_f1", "x_roc_auc", "x_pr_auc"}
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_find_best_threshold_returns_valid_value() -> None:
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    t = find_best_threshold(y, p)
    assert 0.0 < t < 1.0


def test_define_experiments_lists_all_required_models() -> None:
    """CP2 требует ≥4-5 моделей + ансамбли (RF, XGB/LGBM)."""
    names = [c.name for c in define_experiments()]
    assert "LogReg_baseline" in names
    assert "RandomForest_random" in names
    assert "GradientBoosting" in names
    assert "LightGBM_random" in names
    assert len(names) >= 5


def test_results_to_table_sorted_by_f1(splits) -> None:
    """Smoke test: harness работает на маленькой синтетике."""
    X_tr, X_v, X_te, y_tr, y_v, y_te = splits
    # берём только 2 быстрые модели, чтобы тест не висел
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.experiments import ExperimentConfig, _fit_one

    cfg = ExperimentConfig(
        name="logreg_smoke",
        hypothesis="smoke",
        pipeline_factory=lambda: Pipeline(
            [("sc", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=SEED))]
        ),
    )
    result = _fit_one(cfg, X_tr, y_tr, X_v, y_v)
    table = results_to_table([result])
    assert "f1_val" in table.columns
    assert table.iloc[0]["model"] == "logreg_smoke"


def test_select_and_finalize_picks_best_by_f1(splits, tmp_path, monkeypatch) -> None:
    X_tr, X_v, X_te, y_tr, y_v, y_te = splits
    # smoke: один-единственный эксперимент → должен стать финальным
    import src.experiments as experiments_module
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.experiments import ExperimentConfig, _fit_one

    # Не перезаписываем продакшен-артефакт models/final_model.pkl при запуске тестов
    monkeypatch.setattr(experiments_module, "FINAL_MODEL_PATH", tmp_path / "final_model.pkl")

    cfg = ExperimentConfig(
        name="only",
        hypothesis="smoke",
        pipeline_factory=lambda: Pipeline(
            [("sc", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=SEED))]
        ),
    )
    result = _fit_one(cfg, X_tr, y_tr, X_v, y_v)
    final = select_and_finalize([result], X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert final["name"] == "only"
    assert "test_metrics" in final
    assert (tmp_path / "final_model.pkl").exists()
