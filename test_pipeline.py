"""
Run this to test the pipeline: load → validate → preprocess → profile
  python test_pipeline.py
"""
import numpy as np
import pandas as pd

from agent.validator import validate
from agent.preprocessor import preprocess
from agent.profiler import profile_data


def make_test_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        # Normal numeric column
        "age": rng.integers(18, 65, n).astype(float),
        # Numeric with ~15% missing → should get median imputation + indicator
        "income": np.where(rng.random(n) < 0.15, np.nan, rng.normal(50000, 15000, n)),
        # Categorical, clean
        "plan": rng.choice(["free", "basic", "pro"], n),
        # Categorical with ~8% missing → mode imputation + indicator
        "region": np.where(rng.random(n) < 0.08, np.nan, rng.choice(["north", "south", "east", "west"], n)),
        # Target column (binary classification)
        "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
        # Constant column → should be dropped by validator
        "constant": np.ones(n),
        # Column with 90% missing → should be dropped by validator
        "rarely_filled": np.where(rng.random(n) < 0.90, np.nan, rng.random(n)),
    })

    # Inject 15 duplicate rows
    duplicates = df.sample(15, random_state=1)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def divider(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def main():
    df = make_test_data()
    divider(f"Raw data: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.head(3).to_string())

    # Step 1: Validate
    divider("Step 1 — Validate")
    validation = validate(df)
    print(validation.summary())
    print(f"\nUsable columns: {validation.usable_columns}")

    if not validation.passed:
        print("\n[ABORT] Data did not pass validation.")
        return

    # Step 2: Preprocess
    divider("Step 2 — Preprocess")
    preprocess_result = preprocess(df, validation)
    print(preprocess_result.summary())
    print(f"\nDataframe shape after preprocessing: {preprocess_result.df.shape}")

    # Step 3: Profile
    divider("Step 3 — Profile")
    profile = profile_data(preprocess_result.df)
    print(profile.to_llm_summary())


if __name__ == "__main__":
    main()
