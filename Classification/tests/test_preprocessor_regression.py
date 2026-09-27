"""Direct regression test for Decision 1 (Column Alignment Fix).

Verifies that the ColumnTransformer declares passthrough for NUMERICAL_COLS first
and TargetEncoder for COUNTRY_COL second, so that the preprocessed feature matrix
has numeric columns in positions 0..24 and the encoded country in position 25.
This guarantees exact alignment with MONOTONE_VECTOR.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model import (
    COUNTRY_COL,
    FEATURE_COLS,
    MONOTONE_VECTOR,
    NUMERICAL_COLS,
    build_preprocessor,
    create_classifier,
)


def test_preprocessor_column_order_regression(synthetic_train_df: pd.DataFrame):
    """Regression test: verify numeric columns precede encoded country in output matrix."""
    X = synthetic_train_df[FEATURE_COLS]
    y = synthetic_train_df["label"].to_numpy()

    preprocessor = build_preprocessor(cv=5, random_state=42)
    X_trans = preprocessor.fit_transform(X, y)

    # Output matrix must have exactly 26 columns
    assert X_trans.shape == (len(X), 26), f"Expected shape ({len(X)}, 26), got {X_trans.shape}"

    # Verify that the first 25 columns match the exact raw values of NUMERICAL_COLS
    X_num_expected = X[NUMERICAL_COLS].to_numpy(dtype=float)
    X_num_actual = X_trans[:, :25]

    np.testing.assert_allclose(
        X_num_actual,
        X_num_expected,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Decision 1 violation: Columns 0..24 do not match NUMERICAL_COLS!",
    )

    # Verify that column 25 is the target-encoded country column
    country_encoded = X_trans[:, 25]
    # Encoded country should be float values within [0.0, 1.0] (the label space)
    assert np.all(country_encoded >= 0.0) and np.all(country_encoded <= 1.0)
    # Encoded country values should differ from raw country strings
    assert not np.array_equal(country_encoded, X_num_expected[:, 0])

    # Check that transformer output names reflect this exact order
    output_names = preprocessor.get_feature_names_out()
    assert len(output_names) == 26
    for i, col in enumerate(NUMERICAL_COLS):
        assert f"passthrough__{col}" == output_names[i], (
            f"Output feature {i} should be passthrough__{col}, got {output_names[i]}"
        )
    assert output_names[25] == f"target_enc__{COUNTRY_COL[0]}"


def test_monotone_vector_alignment_contract():
    """Verify that MONOTONE_VECTOR length matches FEATURE_COLS and binds to expected features."""
    assert len(MONOTONE_VECTOR) == len(FEATURE_COLS) == 26

    # Verify that first 11 similarity features have +1 constraint
    for i in range(11):
        assert MONOTONE_VECTOR[i] == 1, f"Feature {FEATURE_COLS[i]} should have +1 constraint"

    # Verify that ternary/binary address numeric features have 0 constraint
    assert MONOTONE_VECTOR[11] == 0  # addr_numeric_exact
    assert MONOTONE_VECTOR[12] == 0  # addr_numeric_exists_both

    # Verify retrieval signals have +1 constraint
    assert MONOTONE_VECTOR[13] == 1  # name_bm25_score
    assert MONOTONE_VECTOR[14] == 1  # addr_bm25_score
    assert MONOTONE_VECTOR[15] == 1  # rrf_score
    assert MONOTONE_VECTOR[16] == 1  # stream_overlap_count

    # Verify length ratios and missingness flags have 0 constraint
    for i in range(17, 25):
        assert MONOTONE_VECTOR[i] == 0, f"Feature {FEATURE_COLS[i]} should have 0 constraint"

    # Verify target-encoded country has 0 constraint
    assert MONOTONE_VECTOR[25] == 0
