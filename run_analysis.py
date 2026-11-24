"""Reproduces every table and figure in the README into results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quality.models import components_for, deduplicate, evaluate, load_quality, make_models, top_loadings

OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = 42

if __name__ == "__main__":
    X, y = load_quality()
    n90, ratio = components_for(X, 0.9)
    cum = np.cumsum(ratio)
    pd.DataFrame({"component": range(1, len(ratio) + 1), "explained": ratio, "cumulative": cum}) \
        .to_csv(OUT / "pca_variance.csv", index=False)

    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(range(1, len(cum) + 1), cum, marker="o")
    a.axhline(0.9, color="grey", ls="--")
    a.set_xlabel("principal components")
    a.set_ylabel("cumulative explained variance")
    a.set_title("Red wine chemistry: PCA")
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance.png", dpi=130)
    plt.close(fig)

    # 2D view coloured by quality
    P = PCA(2).fit_transform(StandardScaler().fit_transform(X))
    fig, a = plt.subplots(figsize=(6, 5))
    sc = a.scatter(P[:, 0], P[:, 1], c=y, cmap="viridis", s=10, alpha=0.7)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    fig.colorbar(sc, label="quality score")
    a.set_title("First two PCs, coloured by quality")
    fig.tight_layout()
    fig.savefig(OUT / "pca_2d_by_quality.png", dpi=130)
    plt.close(fig)

    res = evaluate(X, y, make_models(2, n90, SEED), SEED)
    res.insert(0, "data", f"all {len(X)} rows")
    Xd, yd = deduplicate(X, y)
    resd = evaluate(Xd, yd, make_models(2, n90, SEED), SEED)
    resd.insert(0, "data", f"{len(Xd)} unique rows")
    res = pd.concat([res, resd], ignore_index=True)
    res.to_csv(OUT / "model_comparison.csv", index=False)

    print("components for 90%:", n90, "| PC1+PC2:", round(cum[1], 3))
    print("PC1:", top_loadings(StandardScaler().fit_transform(X), list(X.columns), 0, 4))
    print("PC2:", top_loadings(StandardScaler().fit_transform(X), list(X.columns), 1, 4))
    print(y.value_counts().sort_index().to_dict())
    print(res.round(3).to_string(index=False))
