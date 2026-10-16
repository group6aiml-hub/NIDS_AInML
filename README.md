# NIDS with AI/ML on UNSW-NB15

Group 6, AI and Machine Learning, USN.
Goal: detect network intrusions (binary: normal vs. attack) and classify attack type
(multiclass: 10 classes) on the UNSW-NB15 dataset. Main metric: **macro-F1 on the validation set**.

## Team and responsibilities

| Group member | Responsibility |
|---|---|
| Jesper | Data cleaning, preprocessing, Exploratory data anlysis and feature engineering (`02`, `05`, `06`, `src/data.py`) |
| Oliver | Model: <model from lectures> |
| Hubert | Model: <second model> |
| Valentino (+ Jesper) | Model of our own choice: <model> |
| Victoria | Evaluation and report |

## Data

UNSW-NB15 training and testing sets, downloaded from the official UNSW site.
Stored in the group Google Drive under `MyDrive/Datasets/` (not in this repo).

| File | Rows | Made by |
|---|---|---|
| `UNSW_NB15_training-set.csv` | 175 341 | original |
| `UNSW_NB15_testing-set.csv` | 82 332 | original |
| `train_clean.csv` | 107 740 | `02_cleaning.ipynb` |
| `test_clean.csv` | 82 332 | `02_cleaning.ipynb` |

## For model owners: read this before modeling

**1. Always load data through `src/data.py`.** Do not copy preprocessing code between notebooks.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone -q https://github.com/group6aiml-hub/NIDS_AInML.git
import sys; sys.path.append('/content/NIDS_AInML/src')
from data import last_data
d = last_data()
```

See the docstring at the top of `src/data.py` for what `d` contains and a full example.

**2. Rules**
- Never touch the test set. It is used once, in `08_final_test`, on the final chosen setup.
- Compare models on macro-F1 on the validation set (`d['X_val']`). Also report the
  per-class classification report and training time.
- Run both tasks: binary (`y_*_bin`, no SMOTE) and multiclass (`y_*_mc`, with SMOTE).
- Work in your own notebook (e.g. `07a_<model>.ipynb`). Do not edit someone else's notebook,
  since Colab files on the shared Drive overwrite each other.
- If you need a change in the data setup, ask <your name> instead of changing it locally.

## Frozen data decisions

| Decision | Choice | Why / where |
|---|---|---|
| Duplicates | Removed from train (38.6 %), **kept** in test | Avoid leakage between train/val; test should look like real traffic. `02` |
| Contradictory rows | Kept | 1 772 feature groups have more than one label (about 75 % of Analysis and 74 % of Backdoor rows). Limits achievable performance. `02` |
| `id`, `is_ftp_login` | Dropped | `id` is a row number; `is_ftp_login` is identical to `ct_ftp_cmd`. `02` |
| Rare categories | `proto`/`service`/`state` values with < 100 rows → `other` | Counts from train only. `02` |
| Split | 80/20, stratified on `attack_cat`, `random_state=42` | `src/data.py` |
| Preprocessing | One-hot (3 categorical), log1p + StandardScaler (skewed, \|skew\| > 1), StandardScaler (rest) | Fitted on the training part only. `04`, `src/data.py` |
| Class imbalance | SMOTE: Worms/Shellcode/Backdoor/Analysis → 3000, DoS → 5000 (multiclass only) | Best of four methods, but small effect. `05` |
| Features | **TBD** (all 41 / 36 without near-duplicates / ...) | `06` |
| TTL features | **TBD** | Removing them barely changes results. `06` |

## Notebooks (run order)

| Notebook | Content | Status |
|---|---|---|
| `01_explore` | First look at the data | in repo |
| `02_cleaning` | Cleaning, writes `train_clean.csv` / `test_clean.csv` | in repo |
| `03_baseline` | Binary baseline | in repo |
| `04_multiclass` | Multiclass baseline (Random Forest) | in repo |
| `05_imbalance` | No action vs. class_weight vs. SMOTE vs. undersampling | in repo |
| `06_feature_selection` | Correlation, mutual information, RF importance, top-k, with/without TTL | in repo |
| `07_models` | Shared model comparison (LogReg, RF, XGBoost) | in repo |
| `07a`, `07b`, ... | Individual model notebooks | planned |
| `08_final_test` | Final model trained on all of train, evaluated once on test | planned (last step) |

## Results so far (validation, macro-F1)

| Model | Binary | Multiclass |
|---|---|---|
| Logistic Regression | 0.908 | 0.475 |
| Random Forest | 0.932 | 0.572 |
| XGBoost | 0.938 | 0.606 |
