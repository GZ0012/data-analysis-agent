import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    features: list[str]
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    loadings: dict[str, dict[str, float]]   # PC → {feature: loading}
    top_contributors: dict[str, list[str]]   # PC → top 3 features

    def summary(self) -> str:
        lines = [
            f"PCA — Principal Component Analysis",
            f"  Features     : {len(self.features)} columns",
            f"  Components   : {self.n_components}",
            f"",
            f"  Explained Variance:",
        ]
        for i, (ev, cum) in enumerate(zip(self.explained_variance_ratio, self.cumulative_variance)):
            bar = "█" * int(ev * 40)
            lines.append(f"    PC{i+1:<3} {ev:.2%}  (cumulative {cum:.2%})  {bar}")

        lines += [f"", f"  Top Contributing Features per Component:"]
        for pc, contributors in self.top_contributors.items():
            lines.append(f"    {pc}: {', '.join(contributors)}")
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_components: int | None = None,        # None = keep components explaining ≥90% variance
    variance_threshold: float = 0.90,
) -> PCAResult:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First fit full PCA to find how many components hit the variance threshold
    full_pca = PCA().fit(X_scaled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)

    if n_components is None:
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_components = min(n_components, len(feature_cols))

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_.tolist()
    cum = np.cumsum(evr).tolist()

    # Loadings: rows = components, cols = features
    loadings: dict[str, dict[str, float]] = {}
    top_contributors: dict[str, list[str]] = {}

    for i, component in enumerate(pca.components_):
        pc_name = f"PC{i+1}"
        loading_dict = dict(zip(feature_cols, component.tolist()))
        loadings[pc_name] = loading_dict
        top_contributors[pc_name] = sorted(
            feature_cols, key=lambda f: abs(loading_dict[f]), reverse=True
        )[:3]

    return PCAResult(
        features=feature_cols,
        n_components=n_components,
        explained_variance_ratio=evr,
        cumulative_variance=cum,
        loadings=loadings,
        top_contributors=top_contributors,
    )
