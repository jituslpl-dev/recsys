"""Local inference from a trusted, integrity-checked training snapshot."""

from pathlib import Path
import json
import numpy as np
import faiss
import lightgbm as lgb
from recsys.data import sha256
from recsys.features import FeatureBuilder
from recsys.retrieval import FaissRetriever, PopularityRetriever
from recsys.ranking import rerank


class RecommendationService:
    def __init__(self, artifact_directory):
        path = Path(artifact_directory)
        manifest = json.loads((path / "manifest.json").read_text())
        expected = {
            "two_tower.pt",
            "items.faiss",
            "ranker.txt",
            "metadata.json",
            "user_vectors.npy",
            "item_vectors.npy",
            "profiles.npy",
            "activities.npy",
            "popularity.npy",
            "recency.npy",
        }
        if set(manifest) != expected:
            raise ValueError("Artifact manifest does not match required files.")
        for name, digest in manifest.items():
            if Path(name).name != name or sha256(path / name) != digest:
                raise ValueError("Artifact integrity verification failed.")
        metadata = json.loads((path / "metadata.json").read_text())
        self.user_ids = metadata["user_ids"]
        self.item_ids = metadata["item_ids"]
        self.seen = {int(user): set(items) for user, items in metadata["seen"].items()}
        self.candidate_count = metadata["config"]["candidates"]
        users = np.load(path / "user_vectors.npy", allow_pickle=False)
        items = np.load(path / "item_vectors.npy", allow_pickle=False)
        self.features = FeatureBuilder(
            None,
            self.user_ids,
            self.item_ids,
            users,
            items,
            {
                name: np.load(path / f"{name}.npy", allow_pickle=False)
                for name in ["profiles", "activities", "popularity", "recency"]
            },
        )
        if metadata["feature_names"] != self.features.names or users.shape != (
            len(self.user_ids),
            items.shape[1],
        ):
            raise ValueError("Artifact feature dimensions disagree.")
        self.retrieval = FaissRetriever(self.item_ids, items)
        saved_index = faiss.read_index(str(path / "items.faiss"))
        if saved_index.ntotal != len(items) or saved_index.d != items.shape[1]:
            raise ValueError("Saved index dimensions disagree.")
        self.retrieval.index = saved_index
        self.popularity = PopularityRetriever(self.item_ids, self.features.popularity)
        self.ranker = lgb.Booster(model_file=str(path / "ranker.txt"))
        if self.ranker.feature_name() != self.features.names:
            raise ValueError("Saved ranking feature schema disagrees.")

    def recommend(self, user_id, k=10):
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 0:
            raise ValueError("Nonnegative integer user ID required.")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.candidate_count:
            raise ValueError("K must be positive and no larger than the candidate budget.")
        seen = self.seen.get(user_id, set())
        index = self.features.user_map.get(user_id)
        if index is None:
            candidates = self.popularity.search(None, seen, self.candidate_count)
            scored = candidates
            strategy = "cold_user_popularity"
        else:
            candidates = self.retrieval.search(self.features.user_vectors[index], seen, self.candidate_count)
            scored = rerank(self.ranker, self.features, user_id, candidates)
            strategy = "two_tower_faiss_lightgbm"
        return {
            "user_id": user_id,
            "strategy": strategy,
            "candidate_count": len(candidates),
            "recommendations": [{"item_id": item, "score": score} for item, score in scored[:k]],
            "score_note": "Ranking scores are ordering signals, not calibrated probabilities.",
        }
