"""Data loading, PCA structure and cross-validated models for red wine quality.

Every model is a scikit-learn Pipeline, so scaling and PCA are fitted inside
each cross-validation fold (no information leaks from the held-out fold).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "wine_quality_red.csv"


def load_quality(path=None):
    """Red vinho verde: 1143 wines, 11 chemical features, sensory quality 3-8."""
    df = pd.read_csv(path or DATA)
    return df.drop(columns=["quality", "Id"]), df["quality"]


def deduplicate(X, y):
    """Drop repeated measurement rows. The file contains exact duplicates, which
    can land in both the training and the held-out fold and flatter memorising
    models such as random forests."""
    keep = ~X.duplicated().to_numpy()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def components_for(X, threshold=0.9):
    """Smallest number of PCs explaining `threshold` of variance, and the ratios."""
    ratio = make_pipeline(StandardScaler(), PCA()).fit(X).named_steps["pca"].explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ratio), threshold) + 1), ratio


def top_loadings(X, columns, component=0, n=3):
    """Features with the largest absolute weight on a principal component."""
    pca = make_pipeline(StandardScaler(), PCA()).fit(X).named_steps["pca"]
    w = pca.components_[component]
    idx = np.argsort(-np.abs(w))[:n]
    return [(columns[i], float(w[i])) for i in idx]


def make_models(n_pcs_small=2, n_pcs_90=7, seed=42):
    """The models compared, from weakest to strongest assumption."""
    return {
        "majority class": DummyClassifier(strategy="most_frequent"),
        f"logistic on {n_pcs_small} PCs": make_pipeline(
            StandardScaler(), PCA(n_pcs_small), LogisticRegression(max_iter=2000)),
        f"logistic on {n_pcs_90} PCs": make_pipeline(
            StandardScaler(), PCA(n_pcs_90), LogisticRegression(max_iter=2000)),
        "logistic on all 11 features": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "random forest on all 11 features": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
    }


def evaluate(X, y, models, seed=42, folds=5):
    """Mean and std of accuracy over stratified folds, one row per model."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        s = cross_val_score(model, X, y, cv=cv)
        rows.append({"model": name, "cv_accuracy": float(s.mean()), "std": float(s.std())})
    return pd.DataFrame(rows)
