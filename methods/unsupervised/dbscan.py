import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@dataclass
class DBSCANResult:
    features: list[str]
    eps: float
    min_samples: int
    n_clusters: int
    n_noise: int
    noise_ratio: float
    silhouette: float | None   # None if only 1 cluster or all noise
    cluster_sizes: dict[int, int]
    labels: list[int]

    def summary(self) -> str:
        lines = [
            f"DBSCAN Clustering",
            f"  Features     : {len(self.features)} columns",
            f"  eps          : {self.eps}  |  min_samples: {self.min_samples}",
            f"  Clusters found : {self.n_clusters}",
            f"  Noise points   : {self.n_noise} ({self.noise_ratio:.1%} of data)",
        ]
        if self.silhouette is not None:
            lines.append(f"  Silhouette     : {self.silhouette:.4f}")
        else:
            lines.append(f"  Silhouette     : N/A (insufficient clusters)")

        lines += [f"", f"  Cluster Sizes:"]
        total = sum(self.cluster_sizes.values())
        for cid, size in sorted(self.cluster_sizes.items()):
            label = f"Cluster {cid}" if cid >= 0 else "Noise   "
            pct = size / (total + self.n_noise) * 100
            bar = "█" * int(pct / 2)
            lines.append(f"    {label}: {size:>5} samples ({pct:.1f}%)  {bar}")
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DBSCANResult:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum())
    noise_ratio = n_noise / len(labels)

    cluster_sizes = {int(k): int((labels == k).sum()) for k in sorted(unique_labels)}

    # Silhouette only makes sense with ≥2 real clusters and some non-noise points
    silhouette = None
    non_noise_mask = labels != -1
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        silhouette = float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask]))

    return DBSCANResult(
        features=feature_cols,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_ratio=noise_ratio,
        silhouette=silhouette,
        cluster_sizes=cluster_sizes,
        labels=labels.tolist(),
    )
