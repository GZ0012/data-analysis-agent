import pandas as pd
from dataclasses import dataclass, field


MISSING_WARN_THRESHOLD = 0.5   # warn if >50% missing
MISSING_DROP_THRESHOLD = 0.8   # flag as unusable if >80% missing
DUPLICATE_WARN_THRESHOLD = 0.1  # warn if >10% rows are duplicates


@dataclass
class ColumnIssue:
    column: str
    issue: str
    severity: str  # "warn" | "drop"
    suggestion: str


@dataclass
class ValidationResult:
    passed: bool
    fatal_reason: str | None
    warnings: list[str]
    column_issues: list[ColumnIssue]
    duplicate_rows: int
    usable_columns: list[str]

    def summary(self) -> str:
        lines = []
        if not self.passed:
            lines.append(f"[FAIL] {self.fatal_reason}")
            return "\n".join(lines)

        lines.append("[PASS] Data passed validation.")

        if self.duplicate_rows > 0:
            lines.append(f"  - {self.duplicate_rows} duplicate rows detected.")

        for w in self.warnings:
            lines.append(f"  - Warning: {w}")

        for issue in self.column_issues:
            tag = "[DROP]" if issue.severity == "drop" else "[WARN]"
            lines.append(f"  - {tag} '{issue.column}': {issue.issue} → {issue.suggestion}")

        if not self.warnings and not self.column_issues and self.duplicate_rows == 0:
            lines.append("  No issues found.")

        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationResult:
    # Fatal checks
    if df.empty:
        return ValidationResult(
            passed=False,
            fatal_reason="Dataset is empty (0 rows or 0 columns).",
            warnings=[],
            column_issues=[],
            duplicate_rows=0,
            usable_columns=[],
        )

    if len(df.columns) != len(set(df.columns)):
        dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
        return ValidationResult(
            passed=False,
            fatal_reason=f"Duplicate column names found: {list(set(dupes))}",
            warnings=[],
            column_issues=[],
            duplicate_rows=0,
            usable_columns=[],
        )

    warnings = []
    column_issues = []
    columns_to_drop = set()

    # Duplicate rows
    n_duplicates = int(df.duplicated().sum())
    dup_ratio = n_duplicates / len(df)
    if dup_ratio > DUPLICATE_WARN_THRESHOLD:
        warnings.append(
            f"{n_duplicates} duplicate rows ({dup_ratio:.1%} of data). Consider deduplication."
        )

    # Per-column checks
    for col in df.columns:
        series = df[col]
        missing_ratio = series.isna().mean()
        n_unique = series.nunique()

        if missing_ratio > MISSING_DROP_THRESHOLD:
            column_issues.append(ColumnIssue(
                column=col,
                issue=f"{missing_ratio:.1%} missing values",
                severity="drop",
                suggestion="Exclude from analysis — too little data to be useful.",
            ))
            columns_to_drop.add(col)

        elif missing_ratio > MISSING_WARN_THRESHOLD:
            column_issues.append(ColumnIssue(
                column=col,
                issue=f"{missing_ratio:.1%} missing values",
                severity="warn",
                suggestion="Use with caution or apply imputation before modeling.",
            ))

        elif n_unique <= 1:
            column_issues.append(ColumnIssue(
                column=col,
                issue=f"Only {n_unique} unique value(s) — no variance",
                severity="drop",
                suggestion="Exclude from analysis — carries no information.",
            ))
            columns_to_drop.add(col)

    usable_columns = [c for c in df.columns if c not in columns_to_drop]

    if len(usable_columns) == 0:
        return ValidationResult(
            passed=False,
            fatal_reason="No usable columns remain after validation.",
            warnings=warnings,
            column_issues=column_issues,
            duplicate_rows=n_duplicates,
            usable_columns=[],
        )

    return ValidationResult(
        passed=True,
        fatal_reason=None,
        warnings=warnings,
        column_issues=column_issues,
        duplicate_rows=n_duplicates,
        usable_columns=usable_columns,
    )
