"""Shared fixtures and synthetic data generation for Phase 4 tests.

Implements shared test data generator per 04_metrics_logging_testing.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import COUNTRY_COL, FEATURE_COLS, ID_COLS, NUMERICAL_COLS


def generate_synthetic_data(
    n_entities: int = 50,
    cands_per_entity: int = 10,
    random_seed: int = 42,
    include_unseen_country: bool = False,
    singleton_ratio: float = 0.2,
) -> pd.DataFrame:
    """Generate realistic synthetic dataset conforming to Phase 4 input contract.

    Args:
        n_entities: Number of unique S1 entities.
        cands_per_entity: Number of candidates per S1 entity.
        random_seed: Random seed for reproducibility.
        include_unseen_country: If True, include 'france' in some rows.
        singleton_ratio: Proportion of S1 entities with zero true matches.

    Returns:
        pd.DataFrame: Table with s1_id, cand_id, 26 features, and label.
    """
    rng = np.random.default_rng(random_seed)
    total_rows = n_entities * cands_per_entity

    s1_ids: list[str] = []
    cand_ids: list[str] = []
    labels: list[int] = []
    countries: list[str] = []

    country_pool = ["us", "india"]
    if include_unseen_country:
        country_pool.append("france")

    for i in range(n_entities):
        s1_id = f"S1-{i:05d}"
        is_singleton = (rng.random() < singleton_ratio)
        has_positive = False

        entity_country = rng.choice(country_pool)

        for j in range(cands_per_entity):
            cand_id = f"S{rng.choice([2, 3])}-{i * cands_per_entity + j:05d}"
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)
            countries.append(entity_country)

            if is_singleton:
                labels.append(0)
            else:
                # 1 match per entity typically, occasionally 2
                if not has_positive and j == 0:
                    labels.append(1)
                    has_positive = True
                elif has_positive and rng.random() < 0.15:
                    labels.append(1)
                else:
                    labels.append(0)

    # Generate numeric features (0.0 to 1.0)
    data_dict: dict[str, object] = {
        "s1_id": s1_ids,
        "cand_id": cand_ids,
    }

    for col in NUMERICAL_COLS:
        # Give positive labels higher values on similarity features to make it learnable
        base_vals = rng.uniform(0.0, 0.4, size=total_rows)
        pos_boost = np.array(labels) * rng.uniform(0.3, 0.6, size=total_rows)
        feature_vals = np.clip(base_vals + pos_boost, 0.0, 1.0)
        data_dict[col] = feature_vals

    data_dict[COUNTRY_COL[0]] = countries
    data_dict["label"] = labels

    return pd.DataFrame(data_dict)


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Fixture providing a standard synthetic training dataset."""
    return generate_synthetic_data(n_entities=30, cands_per_entity=10, random_seed=42)


@pytest.fixture
def synthetic_test_df_with_france() -> pd.DataFrame:
    """Fixture providing dataset including unseen country 'france'."""
    return generate_synthetic_data(
        n_entities=20,
        cands_per_entity=10,
        random_seed=123,
        include_unseen_country=True,
    )
