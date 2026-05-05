import pandas as pd
from dataclasses import dataclass, field
from agent.validator import ValidationResult


@dataclass
class PreprocessStep:
    column: str
    action: str
    detail: str


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    steps: list[PreprocessStep] = field(default_factory=list)

    def summary(self) -> str:
        if not self.steps:
            return "No preprocessing needed."
        lines = ["Preprocessing steps applied:"]
        for s in self.steps:
            lines.append(f"  - [{s.action}] '{s.column}': {s.detail}")
        return "\n".join(lines)


def preprocess(df: pd.DataFrame, validation: ValidationResult) -> PreprocessResult:
    steps: list[PreprocessStep] = []

    # Work only on usable columns
    df = df[validation.usable_columns].copy()

    # Drop duplicate rows
    if validation.duplicate_rows > 0:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            steps.append(PreprocessStep(
                column="(all rows)",
                action="dedup",
                detail=f"Removed {removed} duplicate rows.",
            ))

    # Handle missing values per column
    for col in df.columns:
        series = df[col]
        n_missing = series.isna().sum()
        if n_missing == 0:
            continue

        missing_ratio = n_missing / len(df)

        # Add indicator column before filling so the missingness signal is preserved
        # Only worth doing if >5% missing — otherwise noise
        if missing_ratio > 0.05:
            indicator_col = f"{col}__was_missing"
            df[indicator_col] = series.isna().astype(int)
            steps.append(PreprocessStep(
                column=col,
                action="add_indicator",
                detail=f"Added '{indicator_col}' to capture missingness as a feature.",
            ))

        # Fill based on column type
        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            df[col] = series.fillna(fill_value)
            steps.append(PreprocessStep(
                column=col,
                action="impute_median",
                detail=f"Filled {n_missing} missing values with median ({fill_value:.4g}).",
            ))

        elif pd.api.types.is_datetime64_any_dtype(series):
            # For datetime, forward-fill then back-fill as a safe default
            df[col] = series.ffill().bfill()
            steps.append(PreprocessStep(
                column=col,
                action="impute_ffill",
                detail=f"Filled {n_missing} missing datetime values via forward/back fill.",
            ))

        else:
            # Categorical / text
            mode_vals = series.mode()
            if len(mode_vals) > 0:
                fill_value = mode_vals[0]
                df[col] = series.fillna(fill_value)
                steps.append(PreprocessStep(
                    column=col,
                    action="impute_mode",
                    detail=f"Filled {n_missing} missing values with mode ('{fill_value}').",
                ))
            else:
                df[col] = series.fillna("Unknown")
                steps.append(PreprocessStep(
                    column=col,
                    action="impute_unknown",
                    detail=f"Filled {n_missing} missing values with 'Unknown'.",
                ))

    return PreprocessResult(df=df, steps=steps)
