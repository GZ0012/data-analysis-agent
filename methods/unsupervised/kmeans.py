import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


@dataclass
class KMeansResult:
    features: list[str]
    n_clusters: int
    inertia: float
    silhouette: float
    davies_bouldin: float
    cluster_sizes: dict[int, int]
    cluster_centers: dict[int, dict[str, float]]  # cluster_id → {feature: center_value}
    labels: list[int]

    def summary(self) -> str:
        lines = [
            f"K-Means Clustering",
            f"  Features   : {len(self.features)} columns",
            f"  K          : {self.n_clusters}",
            f"  Inertia    : {self.inertia:.2f}",
            f"  Silhouette : {self.silhouette:.4f}  (higher is better, max 1.0)",
            f"  Davies-Bouldin : {self.davies_bouldin:.4f}  (lower is better)",
            f"",
            f"  Cluster Sizes:",
        ]
        for cid, size in sorted(self.cluster_sizes.items()):
            pct = size / sum(self.cluster_sizes.values()) * 100
            bar = "█" * int(pct / 2)
            lines.append(f"    Cluster {cid}: {size:>5} samples ({pct:.1f}%)  {bar}")

        lines += [f"", f"  Cluster Centers (original feature scale):"]
        for cid, center in sorted(self.cluster_centers.items()):
            lines.append(f"    Cluster {cid}:")
            for feat, val in center.items():
                lines.append(f"      {feat:<35} {val:.4g}")
        return "\n".join(lines)


def _find_optimal_k(X_scaled: np.ndarray, k_min: int, k_max: int, random_state: int) -> int:
    """Pick k with best silhouette score across candidate values."""
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def run(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int | None = None,   # None = auto-select between 2 and 8
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> KMeansResult:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_clusters is None:
        n_clusters = _find_optimal_k(X_scaled, k_min, k_max, random_state)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X_scaled)

    # Inverse-transform centers back to original scale for readability
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    cluster_centers = {
        int(i): dict(zip(feature_cols, centers_original[i].tolist()))
        for i in range(n_clusters)
    }
    cluster_sizes = {int(k): int((labels == k).sum()) for k in range(n_clusters)}

    return KMeansResult(
        features=feature_cols,
        n_clusters=n_clusters,
        inertia=float(model.inertia_),
        silhouette=float(silhouette_score(X_scaled, labels)),
        davies_bouldin=float(davies_bouldin_score(X_scaled, labels)),
        cluster_sizes=cluster_sizes,
        cluster_centers=cluster_centers,
        labels=labels.tolist(),
    )
