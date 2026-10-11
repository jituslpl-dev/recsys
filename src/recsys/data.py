from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def validate_events(events):
    required = ["user_id", "item_id", "timestamp"]
    if not set(required).issubset(events.columns) or len(events) < 100:
        raise ValueError("At least 100 events with user_id, item_id and timestamp required.")
    result = events[required].copy()
    if result.isna().any().any():
        raise ValueError("Missing event values are not supported.")
    for column in required:
        values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0):
            raise ValueError("Event identifiers/timestamps must be finite nonnegative integers.")
        result[column] = values.astype(np.int64)
    # Earliest event defines first consumption; repeat user-item pairs cannot be future novel targets.
    return result.sort_values(required[::-1], kind="stable").drop_duplicates(
        ["user_id", "item_id"], keep="first"
    )


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    rank_train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cutoffs: tuple[int, int, int]


def temporal_split(events):
    events = validate_events(events)
    stamps = events.timestamp.to_numpy()
    cutoffs = tuple(int(np.quantile(stamps, q, method="inverted_cdf")) for q in [0.6, 0.8, 0.9])
    a, b, c = cutoffs
    if not a < b < c:
        raise ValueError("Too few distinct timestamps for four temporal phases.")
    parts = [
        events[events.timestamp <= a],
        events[(events.timestamp > a) & (events.timestamp <= b)],
        events[(events.timestamp > b) & (events.timestamp <= c)],
        events[events.timestamp > c],
    ]
    if any(len(part) < 10 for part in parts):
        raise ValueError("Each temporal phase must contain at least ten events.")
    return TemporalSplit(*parts, cutoffs)


def history_sets(events):
    return {int(user): set(map(int, group.item_id)) for user, group in events.groupby("user_id")}


def synthetic_events(users=100, items=300, interactions=40, seed=42):
    if users < 10 or items < 20 or not 10 <= interactions < items:
        raise ValueError("Invalid synthetic dataset dimensions.")
    rng = np.random.default_rng(seed)
    groups = np.arange(items) % 10
    rows = []
    for user in range(users):
        preference = user % 10
        probabilities = np.where(groups == preference, 10.0, 1.0)
        selected = rng.choice(items, interactions, replace=False, p=probabilities / probabilities.sum())
        times = np.sort(rng.integers(1, 100000, size=interactions))
        rows.extend((user, int(item), int(time)) for item, time in zip(selected, times))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
