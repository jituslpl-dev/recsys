import numpy as np
import pytest
from recsys.model import sample_negatives, TrainingConfig
from recsys.retrieval import FaissRetriever


def test_faiss_matches_exact_dot_products_and_filters_seen():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=8).astype(np.float32)
    query /= np.linalg.norm(query)
    ids = list(range(100, 140))
    seen = {100, 105, 110}
    result = FaissRetriever(ids, vectors).search(query, seen, 10)
    expected = sorted(
        [(item, float(vectors[i] @ query)) for i, item in enumerate(ids) if item not in seen],
        key=lambda p: (-p[1], p[0]),
    )[:10]
    assert [item for item, _ in result] == [item for item, _ in expected]
    np.testing.assert_allclose([score for _, score in result], [score for _, score in expected], atol=1e-6)
    assert len(FaissRetriever(ids, vectors).search(query, set(ids), 10)) == 0


def test_negative_sampling_excludes_all_known_positives_and_is_seeded():
    positives = {0: {1, 2, 3}, 1: {0, 4}}
    users = np.array([0, 1, 0, 1])
    result = sample_negatives(users, positives, 8, 4, np.random.default_rng(42))
    assert all(value not in positives[int(user)] for user, row in zip(users, result) for value in row)
    np.testing.assert_array_equal(result, sample_negatives(users, positives, 8, 4, np.random.default_rng(42)))
    with pytest.raises(ValueError):
        sample_negatives([0], {0: set(range(8))}, 8, 2, np.random.default_rng(1))


def test_invalid_configuration_and_index():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        FaissRetriever([1, 2], np.ones((2, 8)))
