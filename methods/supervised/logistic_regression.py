import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class LogisticRegressionResult:
    target: str
    features: list[str]
    n_train: int
    n_test: int
    accuracy: float
    f1_weighted: float
    roc_auc: float | None
    coefficients: dict[str, float]
    classes: list[str]
    report: str

    def summary(self) -> str:
        lines = [
            f"Logistic Regression → target: '{self.target}'",
            f"  Features     : {len(self.features)} columns",
            f"  Classes      : {self.classes}",
            f"  Train / Test : {self.n_train} / {self.n_test} samples",
            f"",
            f"  Accuracy     : {self.accuracy:.4f}",
            f"  F1 (weighted): {self.f1_weighted:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"  ROC-AUC      : {self.roc_auc:.4f}")
        lines += [
            f"",
            f"  Coefficients (by absolute impact, on standardized features):",
        ]
        for feat, coef in sorted(self.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {'↑' if coef > 0 else '↓'}  {feat:<35} {coef:+.4f}")
        lines += [f"", f"  Classification Report:", self.report]
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegressionResult:
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    # coef_ shape is (1, n_features) for binary, (n_classes, n_features) for multiclass
    # For multiclass take mean absolute coef per feature as a single importance score
    coef_matrix = model.coef_
    if coef_matrix.shape[0] == 1:
        coef_vals = coef_matrix[0].tolist()
    else:
        coef_vals = coef_matrix.mean(axis=0).tolist()
    coefficients = dict(zip(X.columns.tolist(), coef_vals))

    roc_auc = None
    if len(le.classes_) == 2:
        proba = model.predict_proba(X_test_s)[:, 1]
        roc_auc = float(roc_auc_score(y_test, proba))

    return LogisticRegressionResult(
        target=target_col,
        features=X.columns.tolist(),
        n_train=len(X_train),
        n_test=len(X_test),
        accuracy=float(accuracy_score(y_test, preds)),
        f1_weighted=float(f1_score(y_test, preds, average="weighted")),
        roc_auc=roc_auc,
        coefficients=coefficients,
        classes=[str(c) for c in le.classes_],
        report=classification_report(y_test, preds, target_names=[str(c) for c in le.classes_]),
    )
