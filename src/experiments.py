"""CP2: классификационные эксперименты для предсказания банкротства.

Гоняет 10 конфигураций моделей на одном train/val/test сплите со стратификацией
по таргету и SEED=42, делает GridSearchCV / RandomizedSearchCV для подбора
гиперпараметров, считает PCA-эксперимент, выбирает лучшую модель по F1 на
validation, переобучает на train+val, мерит на test, сохраняет:
  - models/experiments.csv      — таблица для отчёта
  - models/experiments.json     — полные метрики и гиперпараметры
  - models/final_model.pkl      — лучшая модель, переобученная на train+val

Метрика приоритетная — F1 (классы несбалансированны: ~7% bankrupt).
Дополнительно — PR-AUC (надёжна при дисбалансе) и ROC-AUC.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.modeling import MODELS_DIR, clf_metrics, find_best_threshold
from src.preprocessing import SEED, build_dataset, save_processed, split_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENTS_CSV = MODELS_DIR / "experiments.csv"
EXPERIMENTS_JSON = MODELS_DIR / "experiments.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    pipeline_factory: Callable[[], Pipeline]
    param_grid: dict[str, Any] | None = None
    search: str = "grid"
    n_iter: int = 15


@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _evaluate_with_threshold(model: Any, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return clf_metrics(y, pred, prob, prefix="")


def define_experiments() -> list[ExperimentConfig]:
    def logistic_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=SEED, solver="lbfgs"
                    ),
                ),
            ]
        )

    def poly_logistic_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=SEED, solver="lbfgs"
                    ),
                ),
            ]
        )

    def pca_logistic_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=SEED)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=SEED, solver="lbfgs"
                    ),
                ),
            ]
        )

    def knn_pipe() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        )

    def tree_pipe() -> Pipeline:
        return Pipeline(
            [("model", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]
        )

    def rf_pipe() -> Pipeline:
        return Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=SEED, n_jobs=-1
                    ),
                )
            ]
        )

    def gb_pipe() -> Pipeline:
        return Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=400, learning_rate=0.05, max_depth=3, random_state=SEED
                    ),
                )
            ]
        )

    def lgbm_pipe() -> Pipeline:
        return Pipeline(
            [
                (
                    "model",
                    LGBMClassifier(
                        random_state=SEED,
                        n_jobs=-1,
                        verbose=-1,
                        class_weight="balanced",
                    ),
                )
            ]
        )

    return [
        ExperimentConfig(
            name="LogReg_baseline",
            hypothesis="Логистическая регрессия — простой линейный baseline (CP1)",
            pipeline_factory=logistic_pipe,
            param_grid={"model__C": [0.1, 1.0, 10.0]},
            search="grid",
        ),
        ExperimentConfig(
            name="LogReg_poly2",
            hypothesis="Полиномиальные взаимодействия 2-й степени ловят нелинейности",
            pipeline_factory=poly_logistic_pipe,
        ),
        ExperimentConfig(
            name="LogReg_PCA95",
            hypothesis="PCA убирает корреляцию финансовых показателей",
            pipeline_factory=pca_logistic_pipe,
        ),
        ExperimentConfig(
            name="KNN_grid",
            hypothesis="Похожие по профилю компании имеют похожий риск банкротства",
            pipeline_factory=knn_pipe,
            param_grid={
                "model__n_neighbors": [5, 15, 31],
                "model__weights": ["uniform", "distance"],
            },
            search="grid",
        ),
        ExperimentConfig(
            name="DecisionTree_grid",
            hypothesis="Одно дерево даёт интерпретируемые правила",
            pipeline_factory=tree_pipe,
            param_grid={
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_leaf": [10, 50, 200],
            },
            search="grid",
        ),
        ExperimentConfig(
            name="RandomForest_random",
            hypothesis="Bagging-ансамбль усредняет шум, обычно хорош на табличных",
            pipeline_factory=rf_pipe,
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12, 20],
                "model__min_samples_leaf": [1, 5, 20],
                "model__max_features": ["sqrt", 0.5],
            },
            search="random",
            n_iter=10,
        ),
        ExperimentConfig(
            name="GradientBoosting",
            hypothesis="Sequential boosting фиксит ошибки baseline",
            pipeline_factory=gb_pipe,
        ),
        ExperimentConfig(
            name="LightGBM_random",
            hypothesis="Современный GBM — обычно лидер на табличных данных с дисбалансом",
            pipeline_factory=lgbm_pipe,
            param_grid={
                "model__n_estimators": [200, 400, 800],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [-1, 6, 10],
                "model__num_leaves": [31, 63, 127],
                "model__min_child_samples": [20, 50, 100],
            },
            search="random",
            n_iter=15,
        ),
    ]


def _fit_one(
    config: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> ExperimentResult:
    pipeline = config.pipeline_factory()
    started = time.perf_counter()

    if config.param_grid:
        if config.search == "random":
            searcher: Any = RandomizedSearchCV(
                pipeline,
                param_distributions=config.param_grid,
                n_iter=config.n_iter,
                cv=3,
                scoring="f1",
                n_jobs=-1,
                random_state=SEED,
            )
        else:
            searcher = GridSearchCV(
                pipeline,
                param_grid=config.param_grid,
                cv=3,
                scoring="f1",
                n_jobs=-1,
            )
        searcher.fit(X_train, y_train)
        estimator = searcher.best_estimator_
        params = {k: _to_jsonable(v) for k, v in searcher.best_params_.items()}
    else:
        pipeline.fit(X_train, y_train)
        estimator = pipeline
        params = {}

    fit_seconds = time.perf_counter() - started

    prob_val = estimator.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val.values, prob_val)
    metrics = _evaluate_with_threshold(estimator, X_val, y_val, threshold)

    return ExperimentResult(
        name=config.name,
        hypothesis=config.hypothesis,
        estimator=estimator,
        params=params,
        threshold=threshold,
        metrics=metrics,
        fit_seconds=fit_seconds,
    )


def run_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for config in define_experiments():
        print(f"  [{config.name}] {config.hypothesis}")
        result = _fit_one(config, X_train, y_train, X_val, y_val)
        m = result.metrics
        print(
            f"    F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
            f"ROC-AUC={m['roc_auc']:.3f}  thr={result.threshold:.2f}  "
            f"({result.fit_seconds:.1f}s)"
        )
        results.append(result)
    return results


def results_to_table(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "hypothesis": r.hypothesis,
                "best_params": json.dumps(r.params, ensure_ascii=False),
                "threshold": round(r.threshold, 3),
                "f1_val": round(r.metrics["f1"], 4),
                "pr_auc_val": round(r.metrics["pr_auc"], 4),
                "roc_auc_val": round(r.metrics["roc_auc"], 4),
                "precision_val": round(r.metrics["precision"], 4),
                "recall_val": round(r.metrics["recall"], 4),
                "fit_seconds": round(r.fit_seconds, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_val", ascending=False).reset_index(drop=True)


def select_and_finalize(
    results: list[ExperimentResult],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    best = max(results, key=lambda r: r.metrics["f1"])
    print(f"\nЛучшая по F1_val: {best.name} (F1={best.metrics['f1']:.3f})")

    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    final_pipeline = best.estimator
    final_pipeline.fit(X_full, y_full)

    test_metrics = _evaluate_with_threshold(final_pipeline, X_test, y_test, best.threshold)
    print(
        f"Test:  F1={test_metrics['f1']:.3f}  PR-AUC={test_metrics['pr_auc']:.3f}  "
        f"ROC-AUC={test_metrics['roc_auc']:.3f}"
    )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": final_pipeline, "threshold": best.threshold}, f)
    print(f"Финальная модель сохранена -> {FINAL_MODEL_PATH}")

    return {
        "name": best.name,
        "hypothesis": best.hypothesis,
        "params": best.params,
        "threshold": best.threshold,
        "val_metrics": best.metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    print("Загрузка и подготовка датасета...")
    X, y, features = build_dataset()
    print(f"  Признаков использовано: {len(features)} ({features[:3]}...)")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\nЗапуск экспериментов...")
    results = run_all(X_train, y_train, X_val, y_val)

    table = results_to_table(results)
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(EXPERIMENTS_CSV, index=False)
    print(f"\nТаблица экспериментов -> {EXPERIMENTS_CSV}")
    print(table.to_string(index=False))

    final = select_and_finalize(results, X_train, y_train, X_val, y_val, X_test, y_test)

    summary = {
        "seed": SEED,
        "rows": int(len(X)),
        "features": features,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "class_balance": {
            "train_bankrupt_rate": float(y_train.mean()),
            "val_bankrupt_rate": float(y_val.mean()),
            "test_bankrupt_rate": float(y_test.mean()),
        },
        "experiments": [
            {
                "name": r.name,
                "hypothesis": r.hypothesis,
                "params": r.params,
                "threshold": r.threshold,
                "val_metrics": r.metrics,
                "fit_seconds": r.fit_seconds,
            }
            for r in results
        ],
        "final_model": final,
    }
    EXPERIMENTS_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Полные метрики -> {EXPERIMENTS_JSON}")


if __name__ == "__main__":
    main()
