# Bankruptcy Prediction — CP2 work folder

Адаптированная CP2-работа для проекта **«Предсказание банкротства компании по её финансовым показателям»** (hseml-group-project-Yuliasx). Структура и интерфейсы повторяют CP1-репозиторий: `src/preprocessing.py`, `src/modeling.py`, существующие `notebooks/02_baseline.ipynb` и `notebooks/03_experiments.ipynb` использованы как точка старта.

В этой папке:

- `notebooks/cp2.ipynb` — основной CP2-ноутбук с 8 экспериментами, PCA, финальной моделью и feature importance.
- `src/experiments.py` — CLI-оркестратор (`python -m src.experiments`).
- `models/experiments.{csv,json}` + `models/final_model.pkl` — артефакты для отчёта.

---

## Что покрывает CP2 по критериям

| Критерий | Где |
|---|---|
| **4–5 моделей + ансамбли (RF, XGB/LGBM)** | `src/experiments.py::define_experiments` — **8 конфигураций**: LogReg+grid, LogReg+poly2, LogReg+PCA, KNN+grid, DecisionTree+grid, RandomForest+random, GradientBoosting, LightGBM+random |
| **Перебор гиперпараметров** | `GridSearchCV` × 3 модели, `RandomizedSearchCV` × 2 модели, всё с 3-fold CV и `scoring='f1'` |
| **Уменьшение размерности (PCA)** | `LogReg_PCA95` в эксперименте + scree-plot и 2D-визуализация в `cp2.ipynb` |
| **Таблица экспериментов** | `models/experiments.csv` + markdown в README |
| **Финальная модель + обоснование** | `select_and_finalize()` выбирает лучшую по F1-val, переобучает на train+val, сохраняет в `models/final_model.pkl` (вместе с калиброванным порогом) |
| **docker-compose** | `docker-compose.yml` — сервис `notebook` поднимает JupyterLab на :8888 |
| **Метрика приоритетная** | F1 (классы дисбалансированы: ~7% bankrupt, accuracy не информативна) + PR-AUC и ROC-AUC |

---

## Структура папки

```text
.
├── data/
│   ├── raw/
│   │   └── american_bankruptcy.csv     ← датасет (78 682 строки × 22 колонки)
│   └── processed/                       ← X_train/X_val/X_test/y_*.pkl
├── models/
│   ├── experiments.csv                  ← таблица экспериментов
│   ├── experiments.json                 ← полные метрики и гиперпараметры
│   └── final_model.pkl                  ← финальная модель + threshold
├── notebooks/
│   ├── 02_baseline.ipynb                ← из исходного репо (CP1)
│   ├── 03_experiments.ipynb             ← из исходного репо (CP1)
│   └── cp2.ipynb                        ← новый CP2-ноутбук
├── src/
│   ├── preprocessing.py                 ← портирован из CP1, поддержка fyear/year
│   ├── modeling.py                      ← портирован из CP1
│   └── experiments.py                   ← новый: 8 моделей, GridSearch/Random, PCA, финал
├── scripts/
│   └── build_cp2_notebook.py            ← one-shot генератор cp2.ipynb
├── tests/
│   └── test.py                          ← 9 unit-тестов
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                       ← конфиг ruff и pytest
├── requirements.txt
└── README.md
```

---

## Как запустить

### 1. Зависимости

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Проверить, что датасет на месте

```bash
ls data/raw/american_bankruptcy.csv
```

Файл должен быть ~12 МБ. Если нет — скачать с [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset):

```bash
curl -L -o data/raw/american_bankruptcy.csv \
    https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv
```

### 3. Запустить все эксперименты (CLI)

```bash
python -m src.experiments
```

Команда:
- загружает CSV, кодирует таргет (`failed → 1`), удаляет дубли, заполняет NaN, обрезает выбросы по 1%/99%
- стратифицированный сплит train/val/test = 70/15/15 (`SEED=42`)
- сохраняет splits в `data/processed/*.pkl`
- запускает 8 экспериментов, для каждого — 3-fold CV + калибровка порога по F1-val
- пишет таблицу в `models/experiments.csv`, метрики в `models/experiments.json`
- выбирает лучшую по F1-val, переобучает на train+val, мерит на test, сохраняет в `models/final_model.pkl`

### 4. Открыть ноутбук

```bash
jupyter lab
# открыть notebooks/cp2.ipynb и запустить все ячейки сверху вниз
```

### 5. Docker

```bash
docker-compose up --build
# JupyterLab на http://localhost:8888 без токена
```

Запустить эксперименты внутри контейнера:

```bash
docker-compose exec notebook python -m src.experiments
```

---

## Список экспериментов

| # | Имя | Поиск | Гиперпараметры |
|---|---|---|---|
| 1 | LogReg_baseline | Grid | C ∈ {0.1, 1.0, 10.0} |
| 2 | LogReg_poly2 | — | взаимодействия 2-й степени |
| 3 | LogReg_PCA95 | — | PCA до 95% дисперсии |
| 4 | KNN_grid | Grid | n_neighbors ∈ {5,15,31}, weights |
| 5 | DecisionTree_grid | Grid | max_depth, min_samples_leaf |
| 6 | RandomForest_random | Random (10) | n_estimators, max_depth, min_samples_leaf, max_features |
| 7 | GradientBoosting | — | n_estimators=400, lr=0.05, max_depth=3 |
| 8 | LightGBM_random | Random (15) | n_estimators, lr, max_depth, num_leaves, min_child_samples |

Все модели с `class_weight='balanced'` (где поддерживается) — компенсация дисбаланса.
Порог классификации калибруется по F1 на validation в `find_best_threshold()`.

---

## Результаты экспериментов

Validation, отсортировано по F1 (приоритетная метрика):

| Модель | F1 val | PR-AUC val | ROC-AUC val | Precision | Recall | Best params |
|---|---:|---:|---:|---:|---:|---|
| **LightGBM_random** | **0.262** | 0.192 | 0.749 | 0.206 | 0.361 | `n_estimators=800, lr=0.03, num_leaves=31, max_depth=-1, min_child_samples=100` |
| RandomForest_random | 0.241 | 0.174 | 0.733 | 0.169 | 0.420 | `n_estimators=200, max_depth=12, min_samples_leaf=20, max_features=sqrt` |
| KNN_grid | 0.221 | 0.157 | 0.647 | 0.149 | 0.423 | `n_neighbors=5, weights=distance` |
| GradientBoosting | 0.219 | 0.155 | 0.707 | 0.146 | 0.434 | (default 400 / lr=0.05) |
| LogReg_poly2 | 0.187 | 0.118 | 0.646 | 0.119 | 0.433 | — |
| DecisionTree_grid | 0.185 | 0.122 | 0.648 | 0.116 | 0.464 | `max_depth=8, min_samples_leaf=50` |
| LogReg_baseline | 0.178 | 0.113 | 0.628 | 0.115 | 0.395 | `C=0.1` |
| LogReg_PCA95 | 0.178 | 0.114 | 0.629 | 0.114 | 0.396 | — |

### Финальная модель (LightGBM на train+val, оценка на test)

| Стадия | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Validation | 0.262 | 0.192 | 0.749 |
| **Test** | **0.280** | **0.222** | **0.773** |

**Прирост к CP1 baseline (LogReg) на val:** F1 +0.084 (с 0.178 до 0.262, **+47%**), ROC-AUC +0.121 (с 0.628 до 0.749). LightGBM ловит нелинейные взаимодействия финансовых показателей, недоступные линейной регрессии. PCA не помогает — структура важнее сжатия. Деревья без бустинга и KNN бьют линейные методы, но проигрывают LightGBM.

---

## Метрики

**Приоритет — F1.** Классы несбалансированы (~7% banktrupt). Модель «всегда alive» даёт ~93% accuracy, что бесполезно. F1 учитывает обе ошибки на класс банкротов.

Дополнительно:
- **PR-AUC** — threshold-free мера качества для редкого позитивного класса
- **ROC-AUC** — общая ранжирующая способность
- **Precision / Recall** — для понимания trade-off

---

## Тесты

```bash
pytest          # все тесты
ruff check src tests   # линтер
```

В `tests/test.py` 9 unit-тестов:

| Тест | Что проверяет |
|---|---|
| `test_encode_target_creates_binary_column` | `failed → 1`, `alive → 0` |
| `test_drop_duplicates_returns_dataframe` | дубликаты удалены |
| `test_clip_outliers_respects_quantiles` | значения после clip в [q_lower, q_upper] |
| `test_split_keeps_all_rows_and_stratifies` | сплиты в сумме = весь датасет, доля банкротов одинакова |
| `test_clf_metrics_returns_six_keys` | `clf_metrics` возвращает 6 нужных метрик в [0,1] |
| `test_find_best_threshold_returns_valid_value` | порог в (0, 1) |
| `test_define_experiments_lists_all_required_models` | список содержит RF, GBM, LightGBM |
| `test_results_to_table_sorted_by_f1` | smoke: harness работает на маленьком сете |
| `test_select_and_finalize_picks_best_by_f1` | финал выбирает единственную модель |

---

## Зависимости

Python 3.11. Основные пакеты с pinned-версиями (см. `requirements.txt`):

```
numpy 1.26.4    pandas 2.1.3      scikit-learn 1.7.2   lightgbm 4.6.0
joblib 1.5.2    matplotlib 3.10   seaborn 0.13.2
jupyterlab 4.5  ipykernel 7.1     nbformat 5.10
pytest 9.0      ruff 0.15.12
```
