"""
Shared data loading for the NIDS project (UNSW-NB15), group 6.

Every model notebook should use this, so everyone gets the same split,
the same preprocessing and the same SMOTE setup.

Usage in Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !git clone -q https://github.com/group6aiml-hub/NIDS_AInML.git
    import sys; sys.path.append('/content/NIDS_AInML/src')
    from data import last_data
    d = last_data()

    # Example, multiclass with SMOTE:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    p = ImbPipeline([('prep',  d['lag_preprocess']()),
                     ('smote', SMOTE(sampling_strategy=d['smote_strategi'], random_state=42)),
                     ('clf',   YourModel())])
    p.fit(d['X_tr'], d['y_tr_mc'])
    pred = p.predict(d['X_val'])                     # integers 0-9
    d['le'].inverse_transform(pred)                  # back to class names

What you get back (dict d):
    X_tr, X_val            features, 80/20 stratified split of train_clean.csv
    y_tr_txt, y_val_txt    attack_cat as text ('Normal', 'DoS', ...)
    y_tr_mc,  y_val_mc     attack_cat as integers 0-9 (multiclass)
    y_tr_bin, y_val_bin    label, 0 = normal, 1 = attack (binary)
    le                     fitted LabelEncoder, class names in d['le'].classes_
    lag_preprocess         function that returns a NEW, unfitted ColumnTransformer
    smote_strategi         SMOTE sampling_strategy using integer class codes
    cat_cols, skew_cols, other_cols   the column groups used by the preprocessing

Notes / pitfalls:
- y_*_mc are integers 0-9 (required by XGBoost and Keras/MLP).
- smote_strategi uses integer codes, so it only works with y_*_mc.
  If you train on text labels (y_*_txt), use SMOTE_ANTALL instead.
- Call lag_preprocess() once per model. Never share one fitted transformer
  between pipelines.
- For an MLP in Keras: the preprocessing output is a numpy matrix;
  use .astype('float32').
- The test set is NOT loaded here. It is used only once, in 08_final_test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   FunctionTransformer, LabelEncoder)

PATH = '/content/drive/MyDrive/Datasets/'
CAT_COLS = ['proto', 'service', 'state']
FEATURES = None          # None = all 41 features. Set to a list to use a subset.
SMOTE_ANTALL = {'Worms': 3000, 'Shellcode': 3000, 'Backdoor': 3000,
                'Analysis': 3000, 'DoS': 5000}
RANDOM_STATE = 42
SKEW_GRENSE = 1          # |skewness| above this -> log1p before scaling


def last_data(path=PATH):
    train = pd.read_csv(path + 'train_clean.csv')

    X = train.drop(columns=['label', 'attack_cat'])
    if FEATURES is not None:
        X = X[FEATURES]

    X_tr, X_val = train_test_split(X, test_size=0.2,
                                   stratify=train['attack_cat'],
                                   random_state=RANDOM_STATE)

    # Targets in three versions, picked with the same index as X
    y_tr_txt  = train.loc[X_tr.index,  'attack_cat']
    y_val_txt = train.loc[X_val.index, 'attack_cat']

    le = LabelEncoder().fit(train['attack_cat'])
    y_tr_mc  = le.transform(y_tr_txt)
    y_val_mc = le.transform(y_val_txt)

    y_tr_bin  = train.loc[X_tr.index,  'label'].to_numpy()
    y_val_bin = train.loc[X_val.index, 'label'].to_numpy()

    # Column groups. Skewness is computed only on the training part.
    cat_cols   = [c for c in CAT_COLS if c in X_tr.columns]
    num_cols   = [c for c in X_tr.columns if c not in cat_cols]
    skew       = X_tr[num_cols].skew()
    skew_cols  = skew[skew.abs() > SKEW_GRENSE].index.tolist()
    other_cols = [c for c in num_cols if c not in skew_cols]

    def lag_preprocess():
        return ColumnTransformer([
            ('cat',  OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('skew', Pipeline([
                ('log',   FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scale', StandardScaler())]), skew_cols),
            ('num',  StandardScaler(), other_cols),
        ])

    smote_strategi = {int(le.transform([k])[0]): v for k, v in SMOTE_ANTALL.items()}

    return dict(
        X_tr=X_tr, X_val=X_val,
        y_tr_txt=y_tr_txt, y_val_txt=y_val_txt,
        y_tr_mc=y_tr_mc,   y_val_mc=y_val_mc,
        y_tr_bin=y_tr_bin, y_val_bin=y_val_bin,
        le=le,
        cat_cols=cat_cols, skew_cols=skew_cols, other_cols=other_cols,
        lag_preprocess=lag_preprocess,
        smote_strategi=smote_strategi,
    )
