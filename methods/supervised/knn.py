import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class KNNResult:
    target: str
    features: list[str]
    task: str           # "regression" | "classification"
    k: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    cv_score_mean: float
    cv_score_std: float
    report: str | None

    def summary(self) -> str:
        lines = [
            f"KNN ({self.task}) → target: '{self.target}'",
            f"  K (neighbors): {self.k}",
            f"  Features     : {len(self.features)} columns",
            f"  Train / Test : {self.n_train} / {self.n_test} samples",
            f"",
            f"  Metrics:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<20} {v:.4f}")
        lines.append(f"")
        lines.append(f"  Cross-val (5-fold): {self.cv_score_mean:.4f} ± {self.cv_score_std:.4f}")
        if self.report:
            lines += [f"", f"  Classification Report:", self.report]
        return "\n".join(lines)


def _find_optimal_k(X_train, y_train, task: str, k_min: int, k_max: int) -> int:
    """Pick k with best mean cross-val score."""
    best_k, best_score = k_min, -np.inf
    scoring = "r2" if task == "regression" else "f1_weighted"
    for k in range(k_min, k_max + 1):
        model = (
            KNeighborsRegressor(n_neighbors=k)
            if task == "regression"
            else KNeighborsClassifier(n_neighbors=k)
        )
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        if scores.mean() > best_score:
            best_score, best_k = scores.mean(), k
    return best_k


def run(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task: str = "auto",
    k: int | None = None,       # None = auto-select between 3 and 15
    k_min: int = 3,
    k_max: int = 15,
    test_size: float = 0.2,
    random_state: int = 42,
) -> KNNResult:
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    if task == "auto":
        task = "classification" if y.nunique() <= 20 and not pd.api.types.is_float_dtype(y) else "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # KNN is distance-based — scaling is required
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics: dict[str, float] = {}
    report = None

    if task == "regression":
        if k is None:
            k = _find_optimal_k(X_train_s, y_train, task, k_min, k_max)
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        metrics["R²"] = float(r2_score(y_test, preds))
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["MAE"] = float(mean_absolute_error(y_test, preds))
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")

    else:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        if k is None:
            k = _find_optimal_k(X_train_s, y_train_enc, task, k_min, k_max)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_s, y_train_enc)
        preds = model.predict(X_test_s)

        metrics["accuracy"] = float(accuracy_score(y_test_enc, preds))
        metrics["f1_weighted"] = float(f1_score(y_test_enc, preds, average="weighted"))
        if len(le.classes_) == 2:
            proba = model.predict_proba(X_test_s)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test_enc, proba))
        report = classification_report(y_test_enc, preds, target_names=[str(c) for c in le.classes_])
        cv_scores = cross_val_score(model, X_train_s, y_train_enc, cv=5, scoring="f1_weighted")

    return KNNResult(
        target=target_col,
        features=X.columns.tolist(),
        task=task,
        k=k,
        n_train=len(X_train),
        n_test=len(X_test),
        metrics=metrics,
        cv_score_mean=float(cv_scores.mean()),
        cv_score_std=float(cv_scores.std()),
        report=report,
    )
