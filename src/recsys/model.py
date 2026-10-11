from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, users, items, dimension=32):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dimension)
        self.item_embedding = nn.Embedding(items, dimension)
        self.user_network = nn.Sequential(
            nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension)
        )
        self.item_network = nn.Sequential(
            nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension)
        )
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def users(self, indices):
        return F.normalize(self.user_network(self.user_embedding(indices)), dim=-1)

    def items(self, indices):
        return F.normalize(self.item_network(self.item_embedding(indices)), dim=-1)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    embedding_dim: int = 32
    epochs: int = 25
    batch_size: int = 1024
    learning_rate: float = 0.003
    negatives: int = 4
    temperature: float = 0.15

    def __post_init__(self):
        for value, minimum, maximum in [
            (self.embedding_dim, 4, 256),
            (self.epochs, 1, 200),
            (self.batch_size, 16, 8192),
            (self.negatives, 1, 20),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError("Invalid integer training setting.")
        if (
            not np.isfinite([self.learning_rate, self.temperature]).all()
            or not 0 < self.learning_rate <= 0.1
            or not 0 < self.temperature <= 1
        ):
            raise ValueError("Invalid optimizer/temperature setting.")


def sample_negatives(users, positives, item_count, count, rng):
    result = np.empty((len(users), count), dtype=np.int64)
    for row, user in enumerate(users):
        seen = positives[int(user)]
        if len(seen) == item_count:
            raise ValueError("A user has consumed the full catalog; no negative exists.")
        for column in range(count):
            value = int(rng.integers(item_count))
            while value in seen:
                value = int(rng.integers(item_count))
            result[row, column] = value
    return result


def train_two_tower(events, config):
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config.seed)
    user_ids = sorted(map(int, events.user_id.unique()))
    item_ids = sorted(map(int, events.item_id.unique()))
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    users = events.user_id.map(user_map).to_numpy(dtype=np.int64)
    items = events.item_id.map(item_map).to_numpy(dtype=np.int64)
    positives = {index: set(items[users == index]) for index in range(len(user_ids))}
    model = TwoTower(len(user_ids), len(item_ids), config.embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    losses = []
    for epoch in range(config.epochs):
        total = 0.0
        permutation = rng.permutation(len(events))
        for start in range(0, len(events), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            u = torch.from_numpy(users[indices])
            positive = torch.from_numpy(items[indices])
            negative = torch.from_numpy(
                sample_negatives(users[indices], positives, len(item_ids), config.negatives, rng)
            )
            user_vectors = model.users(u)
            positive_scores = (user_vectors * model.items(positive)).sum(dim=-1)
            negative_scores = (user_vectors[:, None, :] * model.items(negative)).sum(dim=-1)
            loss = F.softplus((negative_scores - positive_scores[:, None]) / config.temperature).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append({"epoch": epoch + 1, "pairwise_logistic_loss": total / len(events)})
    model.eval()
    with torch.no_grad():
        user_vectors = model.users(torch.arange(len(user_ids))).numpy().astype(np.float32)
        item_vectors = model.items(torch.arange(len(item_ids))).numpy().astype(np.float32)
    return model, user_ids, item_ids, user_vectors, item_vectors, losses
