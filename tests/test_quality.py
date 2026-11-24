import numpy as np
import pandas as pd
import pytest

from quality.models import (components_for, deduplicate, evaluate, load_quality,
                            make_models, top_loadings)


def test_load_shapes_and_columns():
    X, y = load_quality()
    assert X.shape == (1143, 11) and len(y) == 1143
    assert "quality" not in X.columns and "Id" not in X.columns
    assert set(y) == {3, 4, 5, 6, 7, 8}


def test_deduplicate_removes_repeated_rows_only():
    X = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    y = pd.Series([5, 5, 6])
    Xd, yd = deduplicate(X, y)
    assert len(Xd) == 2 and list(yd) == [5, 6]
    assert not Xd.duplicated().any()


def test_real_data_has_duplicates():
    X, y = load_quality()
    Xd, _ = deduplicate(X, y)
    assert len(Xd) < len(X)


def test_components_for_threshold_monotone():
    X, _ = load_quality()
    n90, ratio = components_for(X, 0.9)
    n50, _ = components_for(X, 0.5)
    assert n50 < n90 <= X.shape[1]
    assert ratio.sum() == pytest.approx(1.0)


def test_top_loadings_sorted_by_magnitude():
    X, _ = load_quality()
    load = top_loadings(X, list(X.columns), 0, 3)
    mags = [abs(w) for _, w in load]
    assert mags == sorted(mags, reverse=True) and len(load) == 3


def test_models_beat_majority_baseline():
    X, y = load_quality()
    Xd, yd = deduplicate(X, y)
    res = evaluate(Xd, yd, make_models(2, 7, 0), seed=0, folds=3).set_index("model")
    base = res.loc["majority class", "cv_accuracy"]
    assert res.loc["logistic on all 11 features", "cv_accuracy"] > base + 0.05
    assert res.loc["random forest on all 11 features", "cv_accuracy"] > base + 0.05


def test_evaluate_is_deterministic():
    X, y = load_quality()
    m = {"lr": make_models(2, 7, 0)["logistic on 2 PCs"]}
    a = evaluate(X, y, m, seed=1)
    b = evaluate(X, y, m, seed=1)
    assert np.allclose(a.cv_accuracy, b.cv_accuracy)
