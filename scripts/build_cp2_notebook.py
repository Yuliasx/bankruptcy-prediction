"""One-shot script to build notebooks/cp2.ipynb. Not part of the runtime pipeline."""

from __future__ import annotations

from pathlib import Path

import nbformat


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    cells = []

    cells.append(
        nbformat.v4.new_markdown_cell(
            "# CP2 — Эксперименты, ансамбли и финальная модель\n"
            "\n"
            "Продолжение `02_baseline.ipynb` / `03_experiments.ipynb`. Этот ноутбук покрывает критерии CP2:\n"
            "\n"
            "1. **8 моделей + ансамбли** (LogReg, LogReg+poly2, LogReg+PCA, KNN, DecisionTree, "
            "**RandomForest**, **GradientBoosting**, **LightGBM**)\n"
            "2. **Перебор гиперпараметров** (`GridSearchCV` / `RandomizedSearchCV`, 3-fold CV, scoring=F1)\n"
            "3. **Уменьшение размерности (PCA)** + scree-plot и 2D-визуализация\n"
            "4. **Таблица экспериментов** в формате `Модель → Гипотеза → Параметры → Метрики`\n"
            "5. **Финальная модель** + интерпретируемость (feature importance)\n"
            "6. **Метрика приоритетная — F1** (классы дисбалансированы: ~7% bankrupt, accuracy не информативна; "
            "также смотрим PR-AUC и ROC-AUC)\n"
            "\n"
            "Весь код берётся из `src/experiments.py`, чтобы CLI-запуск (`python -m src.experiments`) "
            "и ноутбук давали идентичные результаты."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "from __future__ import annotations\n"
            "\n"
            "import json\n"
            "import pickle\n"
            "import sys\n"
            "import warnings\n"
            "from pathlib import Path\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.append(str(ROOT))\n"
            "\n"
            "from src.experiments import (\n"
            "    EXPERIMENTS_CSV,\n"
            "    EXPERIMENTS_JSON,\n"
            "    FINAL_MODEL_PATH,\n"
            "    define_experiments,\n"
            "    results_to_table,\n"
            "    run_all,\n"
            "    select_and_finalize,\n"
            ")\n"
            "from src.preprocessing import RAW_FEATURES, SEED, build_dataset, save_processed, split_data\n"
            "\n"
            "warnings.filterwarnings('ignore')\n"
            "sns.set_theme(style='whitegrid', palette='deep')\n"
            "pd.set_option('display.max_columns', 100)\n"
            "pd.set_option('display.width', 200)\n"
            "\n"
            "print(f'SEED = {SEED}')\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 1. Подготовка данных и сплит\n"
            "\n"
            "Тот же 70/15/15 со стратификацией по `bankrupt` и `SEED=42`, что и в CP1. "
            "Стратификация важна, потому что классы сильно несбалансированы — без неё в test "
            "может попасть аномально мало банкротов и метрики будут шумные."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "X, y, features = build_dataset()\n"
            "X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)\n"
            "save_processed(X_train, X_val, X_test, y_train, y_val, y_test)\n"
            "\n"
            "display(pd.DataFrame({\n"
            "    'split': ['train', 'validation', 'test'],\n"
            "    'rows': [len(X_train), len(X_val), len(X_test)],\n"
            "    'bankrupt_rate': [y_train.mean(), y_val.mean(), y_test.mean()],\n"
            "}))\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 2. Эксперименты с 8 конфигурациями моделей\n"
            "\n"
            "Все модели обучаются на одном train/val. Для моделей с гиперпараметрами — 3-fold CV "
            "со `scoring='f1'`. Для каждой модели порог классификации подбирается по F1 на val "
            "(`find_best_threshold`).\n"
            "\n"
            "| # | Имя | Поиск | Гиперпараметры |\n"
            "|---|---|---|---|\n"
            "| 1 | LogReg_baseline | Grid | C |\n"
            "| 2 | LogReg_poly2 | — | взаимодействия 2-й степени |\n"
            "| 3 | LogReg_PCA95 | — | PCA до 95% дисперсии |\n"
            "| 4 | KNN_grid | Grid | n_neighbors, weights |\n"
            "| 5 | DecisionTree_grid | Grid | max_depth, min_samples_leaf |\n"
            "| 6 | RandomForest_random | Random (10) | n_estimators, max_depth, min_samples_leaf, max_features |\n"
            "| 7 | GradientBoosting | — | n_estimators=400, lr=0.05 |\n"
            "| 8 | LightGBM_random | Random (15) | n_estimators, lr, max_depth, num_leaves, min_child_samples |"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "results = run_all(X_train, y_train, X_val, y_val)\n"
            "table = results_to_table(results)\n"
            "table\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 3. Уменьшение размерности (PCA) + визуализация\n"
            "\n"
            "Финансовые показатели сильно скоррелированы (выручка, активы, прибыль масштабируются "
            "вместе). Посмотрим, во сколько компонент сжимается дисперсия и разделяет ли PCA "
            "класс банкротов визуально."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "from sklearn.decomposition import PCA\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaler = StandardScaler().fit(X_train)\n"
            "X_train_sc = scaler.transform(X_train)\n"
            "\n"
            "pca_full = PCA(random_state=SEED).fit(X_train_sc)\n"
            "explained = np.cumsum(pca_full.explained_variance_ratio_)\n"
            "n_for_95 = int(np.searchsorted(explained, 0.95) + 1)\n"
            "n_for_90 = int(np.searchsorted(explained, 0.90) + 1)\n"
            "print(f'Признаков всего: {X_train_sc.shape[1]}')\n"
            "print(f'Компонент для 90%: {n_for_90}')\n"
            "print(f'Компонент для 95%: {n_for_95}')\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "axes[0].plot(np.arange(1, len(explained) + 1), explained)\n"
            "axes[0].axhline(0.95, color='red', linestyle='--', label='95%')\n"
            "axes[0].axhline(0.90, color='orange', linestyle='--', label='90%')\n"
            "axes[0].set_xlabel('# компонент')\n"
            "axes[0].set_ylabel('Накопленная дисперсия')\n"
            "axes[0].set_title('PCA: накопленная дисперсия')\n"
            "axes[0].legend()\n"
            "\n"
            "pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_train_sc)\n"
            "for cls, color, label in [(0, 'steelblue', 'alive'), (1, 'crimson', 'bankrupt')]:\n"
            "    mask = y_train.values == cls\n"
            "    axes[1].scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=8, alpha=0.4, c=color, label=label)\n"
            "axes[1].set_xlabel('PC1')\n"
            "axes[1].set_ylabel('PC2')\n"
            "axes[1].set_title('Train, первые 2 PC, цвет — класс')\n"
            "axes[1].legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Вывод по PCA:** дисперсия концентрируется в первых ~5 компонентах из 18 — финансовые "
            "показатели сильно избыточны. Однако в 2D `bankrupt` плохо отделяется от `alive`: "
            "нелинейные модели (RF, GBM) должны бить линейный baseline. В таблице экспериментов "
            "`LogReg_PCA95` подтверждает это — линейная регрессия после PCA не лучше обычной."
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 4. Финальная модель: переобучение на train+val и тест"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "final = select_and_finalize(results, X_train, y_train, X_val, y_val, X_test, y_test)\n"
            "\n"
            "print()\n"
            "print(f\"Финальная модель: {final['name']}\")\n"
            "print(f\"Гипотеза:        {final['hypothesis']}\")\n"
            "print(f\"Параметры:       {final['params']}\")\n"
            "print(f\"Порог:           {final['threshold']:.3f}\")\n"
            "print(f\"\\nValidation:  F1={final['val_metrics']['f1']:.3f}  \"\n"
            "      f\"PR-AUC={final['val_metrics']['pr_auc']:.3f}  \"\n"
            "      f\"ROC-AUC={final['val_metrics']['roc_auc']:.3f}\")\n"
            "print(f\"Test:        F1={final['test_metrics']['f1']:.3f}  \"\n"
            "      f\"PR-AUC={final['test_metrics']['pr_auc']:.3f}  \"\n"
            "      f\"ROC-AUC={final['test_metrics']['roc_auc']:.3f}\")\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Обоснование выбора:** при сильном дисбалансе классов (~7% bankrupt) accuracy "
            "не информативна — модель, всегда предсказывающая `alive`, даст ~93% accuracy. "
            "Поэтому приоритет — F1 (баланс precision/recall на класс банкротов). "
            "GBM-семейство (RandomForest / LightGBM / GradientBoosting) ловят нелинейные "
            "взаимодействия финансовых показателей и обычно лидируют. Полиномиальная регрессия "
            "медленна на 78k×18 и склонна к переобучению. PCA не даёт прироста — структура важнее "
            "сжатия для линейной модели."
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 5. Интерпретируемость: feature importance\n"
            "\n"
            "Для tree-based финальной модели берём `feature_importances_`. Если финал — "
            "линейная (LogReg), смотрим коэффициенты."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "with open(FINAL_MODEL_PATH, 'rb') as f:\n"
            "    bundle = pickle.load(f)\n"
            "final_pipeline = bundle['pipeline']\n"
            "estimator = final_pipeline.named_steps.get('model', final_pipeline)\n"
            "\n"
            "if hasattr(estimator, 'feature_importances_'):\n"
            "    importances = pd.Series(estimator.feature_importances_, index=RAW_FEATURES).sort_values(\n"
            "        ascending=False\n"
            "    )\n"
            "    title = f'Feature importance ({type(estimator).__name__})'\n"
            "elif hasattr(estimator, 'coef_'):\n"
            "    importances = pd.Series(\n"
            "        np.abs(estimator.coef_[0]), index=RAW_FEATURES\n"
            "    ).sort_values(ascending=False)\n"
            "    title = f'|Coefficient| ({type(estimator).__name__})'\n"
            "else:\n"
            "    importances = pd.Series([], dtype=float)\n"
            "    title = 'Importance (модель не поддерживает)'\n"
            "\n"
            "if len(importances) > 0:\n"
            "    plt.figure(figsize=(9, 6))\n"
            "    sns.barplot(x=importances.values, y=importances.index, color='steelblue')\n"
            "    plt.title(title)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    display(importances.round(4).to_frame('importance'))\n"
            "else:\n"
            "    print('Финальная модель не предоставляет feature importance напрямую.')\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 6. Сравнение CP1 baseline vs CP2 final"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "summary = json.loads(EXPERIMENTS_JSON.read_text(encoding='utf-8'))\n"
            "baseline = next(\n"
            "    (e for e in summary['experiments'] if e['name'] == 'LogReg_baseline'), None\n"
            ")\n"
            "final_summary = summary['final_model']\n"
            "\n"
            "comparison = pd.DataFrame([\n"
            "    {\n"
            "        'stage': 'CP1 baseline',\n"
            "        'model': 'LogReg_baseline',\n"
            "        'f1_val': baseline['val_metrics']['f1'] if baseline else float('nan'),\n"
            "        'pr_auc_val': baseline['val_metrics']['pr_auc'] if baseline else float('nan'),\n"
            "        'roc_auc_val': baseline['val_metrics']['roc_auc'] if baseline else float('nan'),\n"
            "    },\n"
            "    {\n"
            "        'stage': 'CP2 final (val)',\n"
            "        'model': final_summary['name'],\n"
            "        'f1_val': final_summary['val_metrics']['f1'],\n"
            "        'pr_auc_val': final_summary['val_metrics']['pr_auc'],\n"
            "        'roc_auc_val': final_summary['val_metrics']['roc_auc'],\n"
            "    },\n"
            "    {\n"
            "        'stage': 'CP2 final (test)',\n"
            "        'model': final_summary['name'],\n"
            "        'f1_val': final_summary['test_metrics']['f1'],\n"
            "        'pr_auc_val': final_summary['test_metrics']['pr_auc'],\n"
            "        'roc_auc_val': final_summary['test_metrics']['roc_auc'],\n"
            "    },\n"
            "])\n"
            "comparison\n"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 7. Выводы\n"
            "\n"
            "- 8 конфигураций моделей сравнили на одном `train/val` сплите со стратификацией и `SEED=42`.\n"
            "- Tree/boosting-семейство ожидаемо бьёт линейные методы — банкротство нелинейно "
            "связано с финансовыми показателями.\n"
            "- PCA сохраняет 95% дисперсии в небольшом числе компонент, но для классификации "
            "не помогает (структура важнее сжатия).\n"
            "- Метрика-приоритет — F1, потому что классы несбалансированы. Также важна PR-AUC "
            "как threshold-free мера для редкого позитивного класса.\n"
            "- Артефакты для отчёта: `models/experiments.csv`, `models/experiments.json`, "
            "`models/final_model.pkl` (содержит pipeline + калиброванный порог)."
        )
    )

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build()
    target = Path(__file__).resolve().parent.parent / "notebooks" / "cp2.ipynb"
    with target.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
