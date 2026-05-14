import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder


@dataclass
class RandomForestResult:
    target: str
    features: list[str]
    task: str           # "regression" | "classification"
    n_estimators: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    feature_importance: dict[str, float]   # sorted descending
    report: str | None

    def summary(self) -> str:
        lines = [
            f"Random Forest ({self.task}) → target: '{self.target}'",
            f"  Trees        : {self.n_estimators}",
            f"  Features     : {len(self.features)} columns",
            f"  Train / Test : {self.n_train} / {self.n_test} samples",
            f"",
            f"  Metrics:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<20} {v:.4f}")
        lines += [f"", f"  Feature Importance (top 10):"]
        for feat, imp in list(self.feature_importance.items())[:10]:
            bar = "█" * int(imp * 40)
            lines.append(f"    {feat:<35} {imp:.4f}  {bar}")
        if self.report:
            lines += [f"", f"  Classification Report:", self.report]
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task: str = "auto",
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RandomForestResult:
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

    metrics: dict[str, float] = {}
    report = None

    if task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics["R²"] = float(r2_score(y_test, preds))
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["MAE"] = float(mean_absolute_error(y_test, preds))

    else:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train_enc)
        preds = model.predict(X_test)

        metrics["accuracy"] = float(accuracy_score(y_test_enc, preds))
        metrics["f1_weighted"] = float(f1_score(y_test_enc, preds, average="weighted"))
        if len(le.classes_) == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test_enc, proba))
        report = classification_report(y_test_enc, preds, target_names=[str(c) for c in le.classes_])

    importance_sorted = dict(
        sorted(
            zip(X.columns.tolist(), model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return RandomForestResult(
        target=target_col,
        features=X.columns.tolist(),
        task=task,
        n_estimators=n_estimators,
        n_train=len(X_train),
        n_test=len(X_test),
        metrics=metrics,
        feature_importance=importance_sorted,
        report=report,
    )
