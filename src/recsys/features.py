import numpy as np


class FeatureBuilder:
    """Feature statistics and histories are frozen at retrieval-training cutoff."""

    def __init__(self, train, user_ids, item_ids, user_vectors, item_vectors, statistics=None):
        self.user_map = {int(value): i for i, value in enumerate(user_ids)}
        self.item_map = {int(value): i for i, value in enumerate(item_ids)}
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        if statistics is not None:
            for name, shape in [
                ("popularity", (len(item_ids),)),
                ("recency", (len(item_ids),)),
                ("activities", (len(user_ids),)),
                ("profiles", user_vectors.shape),
            ]:
                value = np.asarray(statistics[name])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Invalid frozen feature statistics.")
                setattr(self, name, value)
            return
        self.popularity = np.array([int((train.item_id == item).sum()) for item in item_ids], dtype=float)
        self.activities = np.zeros(len(user_ids), dtype=float)
        self.profiles = np.zeros_like(user_vectors)
        self.recency = np.zeros(len(item_ids), dtype=float)
        span = max(1, int(train.timestamp.max() - train.timestamp.min()))
        for user, group in train.groupby("user_id"):
            index = self.user_map[int(user)]
            self.activities[index] = len(group)
            profile = item_vectors[[self.item_map[int(item)] for item in group.item_id]].mean(axis=0)
            self.profiles[index] = profile / max(np.linalg.norm(profile), 1e-12)
        for item, group in train.groupby("item_id"):
            self.recency[self.item_map[int(item)]] = (
                int(train.timestamp.max()) - int(group.timestamp.max())
            ) / span

    @property
    def names(self):
        d = self.item_vectors.shape[1]
        return (
            [
                "retrieval_cosine",
                "log_training_popularity",
                "log_user_training_activity",
                "history_affinity",
                "training_relative_age",
            ]
            + [f"interaction_{i}" for i in range(d)]
            + [f"item_vector_{i}" for i in range(d)]
            + [f"user_vector_{i}" for i in range(d)]
        )

    def build(self, user, candidates):
        indices = np.array([self.item_map[int(item)] for item, _ in candidates], dtype=int)
        if not len(indices):
            return np.empty((0, len(self.names)))
        uid = self.user_map.get(int(user))
        vector = np.zeros(self.item_vectors.shape[1]) if uid is None else self.user_vectors[uid]
        profile = np.zeros_like(vector) if uid is None else self.profiles[uid]
        activity = 0 if uid is None else self.activities[uid]
        vectors = self.item_vectors[indices]
        basic = np.column_stack(
            [
                [score for _, score in candidates],
                np.log1p(self.popularity[indices]),
                np.full(len(indices), np.log1p(activity)),
                vectors @ profile,
                self.recency[indices],
            ]
        )
        return np.column_stack([basic, vectors * vector, vectors, np.tile(vector, (len(indices), 1))])
