import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str  # "numeric", "categorical", "datetime", "text"
    n_missing: int
    missing_pct: float
    n_unique: int
    sample_values: list[Any]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]

    def to_llm_summary(self) -> str:
        lines = [
            f"Dataset: {self.n_rows} rows × {self.n_cols} columns",
            "",
            "Columns:",
        ]
        for col in self.columns:
            stat_str = ""
            if col.kind == "numeric" and col.stats:
                stat_str = (
                    f" | mean={col.stats.get('mean', 'N/A'):.3g}"
                    f", std={col.stats.get('std', 'N/A'):.3g}"
                    f", range=[{col.stats.get('min', 'N/A'):.3g}, {col.stats.get('max', 'N/A'):.3g}]"
                )
            elif col.kind == "binary_indicator":
                stat_str = f" | flagged={col.stats.get('flagged_count', 0)} rows"
            elif col.kind == "categorical" and col.stats:
                top = list(col.stats.get("top_values", {}).keys())[:3]
                stat_str = f" | top_values={top}"

            lines.append(
                f"  - {col.name} [{col.kind}]"
                f" | unique={col.n_unique}"
                f"{stat_str}"
                f" | samples={col.sample_values[:3]}"
            )
        return "\n".join(lines)


def profile_data(df: pd.DataFrame) -> DataProfile:
    columns = []
    for col_name in df.columns:
        series = df[col_name]
        # Post-preprocessing: missing values should already be filled
        n_missing = int(series.isna().sum())
        missing_pct = n_missing / len(series) * 100
        n_unique = int(series.nunique())

        # Indicator columns added by preprocessor are binary flags, not general numerics
        if col_name.endswith("__was_missing"):
            kind = "binary_indicator"
            stats = {"flagged_count": int(series.sum())}

        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
            stats = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "datetime"
            stats = {"min": str(series.min()), "max": str(series.max())}
        elif n_unique / max(len(series), 1) < 0.05 or n_unique <= 20:
            kind = "categorical"
            stats = {"top_values": series.value_counts().head(5).to_dict()}
        else:
            kind = "text"
            stats = {}

        sample_values = series.dropna().head(5).tolist()

        columns.append(
            ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                kind=kind,
                n_missing=n_missing,
                missing_pct=missing_pct,
                n_unique=n_unique,
                sample_values=sample_values,
                stats=stats,
            )
        )

    return DataProfile(n_rows=len(df), n_cols=len(df.columns), columns=columns)
