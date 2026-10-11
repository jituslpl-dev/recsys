import numpy as np
import pandas as pd
import pytest
from recsys.metrics import user_metrics, evaluate
from recsys.data import temporal_split, synthetic_events, validate_events


def test_known_metric_values_and_macro_averaging():
    result = user_metrics([8, 2, 3], {2, 3, 9}, 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["mrr"] == 0.5
    assert result["hit_rate"] == 1
    assert result["ndcg"] == pytest.approx(
        (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    )
    summary = evaluate({1: [4], 2: []}, {1: {4}, 2: {4}}, [1])
    assert summary["metrics"]["@1"]["recall"] == 0.5
    with pytest.raises(ValueError):
        user_metrics([2, 2], {2}, 2)


def test_zero_hit_and_short_recommendations():
    assert user_metrics([], {5}, 10) == {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "hit_rate": 0.0}
    assert user_metrics([5], {5, 6}, 10)["recall"] == 0.5


def test_global_time_boundaries_and_no_pair_overlap():
    events = synthetic_events()
    split = temporal_split(events)
    parts = [split.train, split.rank_train, split.validation, split.test]
    assert all(left.timestamp.max() < right.timestamp.min() for left, right in zip(parts[:-1], parts[1:]))
    pairs = [set(zip(p.user_id, p.item_id)) for p in parts]
    assert all(not pairs[a] & pairs[b] for a in range(4) for b in range(a + 1, 4))


def test_equal_timestamps_never_cross_phases_and_earliest_duplicate():
    events = synthetic_events()
    events.timestamp = (events.timestamp // 1000) * 1000
    duplicated = pd.concat([events, events.iloc[:1].assign(timestamp=200000)])
    clean = validate_events(duplicated)
    assert len(clean) == len(events)
    split = temporal_split(duplicated)
    parts = [split.train, split.rank_train, split.validation, split.test]
    assert all(set(left.timestamp).isdisjoint(right.timestamp) for left, right in zip(parts[:-1], parts[1:]))


def test_invalid_event_contracts():
    events = synthetic_events()
    events.loc[0, "item_id"] = -1
    with pytest.raises(ValueError):
        temporal_split(events)
    with pytest.raises(ValueError):
        temporal_split(pd.DataFrame({"user_id": range(100), "item_id": range(100), "timestamp": [1] * 100}))
