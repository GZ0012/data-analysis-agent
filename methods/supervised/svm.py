import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class SVMResult:
    target: str
    features: list[str]
    task: str          # "regression" | "classification"
    kernel: str
    n_train: int
    n_test: int
    metrics: dict[str, float]
    classification_report: str | None = None

    def summary(self) -> str:
        lines = [
            f"SVM ({self.task}) → target: '{self.target}'",
            f"  Kernel   : {self.kernel}",
            f"  Features : {len(self.features)} columns",
            f"  Train / Test : {self.n_train} / {self.n_test} samples",
            f"",
            f"  Metrics:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<20} {v:.4f}")
        if self.classification_report:
            lines.append(f"\n  Classification Report:\n{self.classification_report}")
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task: str = "auto",        # "auto" | "regression" | "classification"
    kernel: str = "rbf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SVMResult:
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    # Infer task from target dtype
    if task == "auto":
        task = "classification" if y.nunique() <= 20 and not pd.api.types.is_float_dtype(y) else "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics: dict[str, float] = {}
    report = None

    if task == "regression":
        model = SVR(kernel=kernel)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        metrics["R²"] = float(r2_score(y_test, preds))
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["MAE"] = float(mean_absolute_error(y_test, preds))

    else:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        model = SVC(kernel=kernel, probability=True, random_state=random_state)
        model.fit(X_train_s, y_train_enc)
        preds = model.predict(X_test_s)

        metrics["accuracy"] = float(accuracy_score(y_test_enc, preds))
        metrics["f1_weighted"] = float(f1_score(y_test_enc, preds, average="weighted"))
        if len(le.classes_) == 2:
            proba = model.predict_proba(X_test_s)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test_enc, proba))
        report = classification_report(y_test_enc, preds, target_names=[str(c) for c in le.classes_])

    return SVMResult(
        target=target_col,
        features=X.columns.tolist(),
        task=task,
        kernel=kernel,
        n_train=len(X_train),
        n_test=len(X_test),
        metrics=metrics,
        classification_report=report,
    )
