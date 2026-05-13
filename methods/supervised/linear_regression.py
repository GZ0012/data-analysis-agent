import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearRegressionResult:
    target: str
    features: list[str]
    r2_train: float
    r2_test: float
    rmse_test: float
    mae_test: float
    coefficients: dict[str, float]
    intercept: float
    n_train: int
    n_test: int

    def summary(self) -> str:
        lines = [
            f"Linear Regression → target: '{self.target}'",
            f"  Features : {len(self.features)} columns",
            f"  Train / Test : {self.n_train} / {self.n_test} samples",
            f"",
            f"  R² (train)  : {self.r2_train:.4f}",
            f"  R² (test)   : {self.r2_test:.4f}",
            f"  RMSE (test) : {self.rmse_test:.4f}",
            f"  MAE  (test) : {self.mae_test:.4f}",
            f"",
            f"  Coefficients (by absolute impact, on standardized features):",
        ]
        for feat, coef in sorted(self.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {'↑' if coef > 0 else '↓'}  {feat:<35} {coef:+.4f}")
        lines.append(f"    intercept{'':<35} {self.intercept:+.4f}")
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> LinearRegressionResult:
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    return LinearRegressionResult(
        target=target_col,
        features=X.columns.tolist(),
        r2_train=float(r2_score(y_train, model.predict(X_train_s))),
        r2_test=float(r2_score(y_test, model.predict(X_test_s))),
        rmse_test=float(np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))),
        mae_test=float(mean_absolute_error(y_test, model.predict(X_test_s))),
        coefficients=dict(zip(X.columns.tolist(), model.coef_.tolist())),
        intercept=float(model.intercept_),
        n_train=len(X_train),
        n_test=len(X_test),
    )
