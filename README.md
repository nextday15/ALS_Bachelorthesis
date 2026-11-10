# Bachelor Thesis: Machine Learning-Based Classification of ALS Dysarthria Severity Using Acoustic Features

5-class severity prediction from SAND Task 1 speech. The primary metric is the unweighted mean of per-fold Macro-F1.

## Data

- Eight recordings per speaker: vowels `/a,e,i,o,u/` and DDK `/pa,ta,ka/`
- Native audio is 8 kHz, mono, 16-bit. No resampling, no duration padding
- Person-level `StratifiedGroupKFold` by speaker ID (k=5)
- Labels and splits: `data/raw/task1_training/sand_task_1.xlsx`

## Features

- eGeMAPSv02 functionals: 88 × 8 tasks = **704**
- Extra DDK (`ddk_rate`, `pause_ratio`, `pvi` on `/pa,ta,ka/`) + VSA from `/i,a,u/` F1–F2 = **10**
- Fused matrix: **219 × 714** (`fused_egemaps_ddk_vsa_train`)
- Praat formants use `maximum_formant=3800` (Nyquist 4 kHz)

## Models and preprocessing

SVM-RBF, Random Forest, XGBoost, ordinal logit (`mord.LogisticAT`). Fixed baseline hyperparameters (`random_state=42`). SVM/RF use `class_weight="balanced"`; XGB uses balanced `sample_weight`. No class resampling.

In-fold only: `inf → NaN` → median impute → optional ANOVA / Pearson → `StandardScaler`.

Feature selection also in-fold:

- ANOVA: keep features with `f_classif` F ≥ 4
- Pearson: drop pairs with `|r| > 0.9`, keep the higher-F feature

## Experiments so far

Mean Macro-F1, averaged across folds (`outputs/tables/result_comparison.csv`):

| experiment | n_features | svm | rf | xgb | ordinal |
|---|---|---|---|---|---|
| baseline (eGeMAPS) | 704 | 0.266 | 0.271 | 0.378 | 0.359 |
| rhythm (+ DDK/VSA) | 714 | 0.270 | 0.295 | 0.366 | 0.355 |
| anova (F ≥ 4) | 123.8 | 0.376 | 0.302 | 0.374 | 0.418 |
| pearson (F ≥ 4 then \|r\| > 0.9) | 106.6 | 0.399 | 0.315 | 0.355 | 0.388 |

`n_features` for anova/pearson is the 5-fold mean.

## Run

From the repo root, with `PYTHONPATH=.` (Anaconda Python with `sklearn`, `xgboost`, `mord`, `opensmile`, `parselmouth`):

```bash
python -m src.features.extract_eGeMAPS
python -m src.features.extract_rhythm
python -m src.features.build_matrices
python -m src.models.train_baseline
python -m src.evaluation.corr_nfeat_curve
python -m src.evaluation.anova_f_curve
```

Switch the active CV run in `src/models/train_baseline.py` via `EXPERIMENT`.

## Layout

```
src/data_loader.py          # labels, audio inventory, SGKF
src/features/               # extraction, 714-d fusion, ANOVA/Pearson filters
src/models/                 # pipelines and CV
src/evaluation/             # metrics, class-distribution plot, filter curves
outputs/processed_data/     # long tables, fused matrices, OOF predictions
outputs/tables/             # CV results; 0_baseline, 1.1 Rhythm, 2.1 Anova, 3.1 Pearson
outputs/figures/            # class_distribution, corr_nfeat_curve, anova_f_curve
```

## TODO

- Hyperparameter optimization using `GridSearchCV`
- Feature group ablation study
- SHAP-based feature importance analysis
- Final model evaluation on the held-out validation set
