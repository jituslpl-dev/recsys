from typing import Protocol
import numpy as np
import faiss


class CandidateRetriever(Protocol):
    def search(self, user_vector, seen, count): ...


class FaissRetriever:
    """Exact inner-product retrieval over normalized neural item embeddings."""

    def __init__(self, item_ids, item_vectors):
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.item_ids) or len(self.vectors) == 0:
            raise ValueError("Item IDs must match a nonempty embedding matrix.")
        if len(set(self.item_ids)) != len(self.item_ids) or not np.isfinite(self.vectors).all():
            raise ValueError("Unique item identifiers and finite embeddings required.")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError("FAISS cosine retrieval requires normalized item embeddings.")
        faiss.omp_set_num_threads(2)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, user_vector, seen, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Candidate count must be a positive integer.")
        vector = np.asarray(user_vector, dtype=np.float32)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Query must match finite embedding dimensions.")
        if not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("Query must be normalized.")
        seen = set(seen)
        limit = min(len(self.item_ids), count + len(seen))
        scores, indices = self.index.search(np.ascontiguousarray(vector[None, :]), limit)
        pairs = [
            (int(self.item_ids[index]), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0 and int(self.item_ids[index]) not in seen
        ]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[:count]


class PopularityRetriever:
    def __init__(self, item_ids, popularity):
        self.ordered = sorted(
            zip(map(int, item_ids), map(float, popularity)), key=lambda pair: (-pair[1], pair[0])
        )

    def search(self, user_vector, seen, count):
        return [pair for pair in self.ordered if pair[0] not in seen][:count]
