from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import faiss
import lightgbm
from recsys.data import temporal_split, history_sets, sha256
from recsys.model import TrainingConfig, train_two_tower
from recsys.retrieval import FaissRetriever, PopularityRetriever
from recsys.features import FeatureBuilder
from recsys.ranking import fit_ranker, rerank
from recsys.metrics import evaluate


def candidate_groups(targets, histories, user_map, user_vectors, retrieval, popularity, count):
    groups = []
    for user, group in targets.groupby("user_id"):
        user = int(user)
        seen = histories.get(user, set())
        relevant = set(map(int, group.item_id)) - seen
        if not relevant:
            continue
        index = user_map.get(user)
        if index is None:
            candidates = popularity.search(None, seen, count)
        else:
            candidates = retrieval.search(user_vectors[index], seen, count)
        groups.append((user, candidates, relevant))
    return groups


def benchmark(groups, history, user_map, ranker, features, popularity, ks, count):
    relevance, retrieved, ranked, popular = {}, {}, {}, {}
    latencies = []
    for user, candidates, targets in groups:
        relevance[user] = targets
        retrieved[user] = [item for item, _ in candidates]
        start = time.perf_counter()
        scored = candidates if user not in user_map else rerank(ranker, features, user, candidates)
        latencies.append((time.perf_counter() - start) * 1000)
        ranked[user] = [item for item, _ in scored]
        popular[user] = [item for item, _ in popularity.search(None, history.get(user, set()), count)]
    all_results = {
        "retrieval": evaluate(retrieved, relevance, ks),
        "two_stage": evaluate(ranked, relevance, ks),
        "popularity": evaluate(popular, relevance, ks),
        "candidate_recall": evaluate(retrieved, relevance, [count]),
        "ranker_only_local_ms": {
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
        },
        "cohorts": {},
    }
    for name, users in [
        ("warm_users", set(relevance) & set(user_map)),
        ("cold_users", set(relevance) - set(user_map)),
    ]:
        if users:
            rel = {user: relevance[user] for user in sorted(users)}
            all_results["cohorts"][name] = {
                "retrieval": evaluate(retrieved, rel, ks),
                "two_stage": evaluate(ranked, rel, ks),
                "popularity": evaluate(popular, rel, ks),
            }
    catalog = set(features.item_map)
    all_results["out_of_catalog_target_fraction"] = sum(
        len(target - catalog) for target in relevance.values()
    ) / sum(map(len, relevance.values()))
    all_results["catalog_coverage_at_20"] = len(
        {item for items in ranked.values() for item in items[:20]}
    ) / len(catalog)
    return all_results, [
        {
            "user_id": user,
            "relevant_item_ids": sorted(relevance[user]),
            "retrieved": retrieved[user],
            "ranked": ranked[user],
            "popularity": popular[user],
        }
        for user in sorted(relevance)
    ]


def run(events, output, config=None, source=None):
    config = config or {}
    training = TrainingConfig(**{k: v for k, v in config.items() if k in TrainingConfig.__dataclass_fields__})
    count = config.get("candidates", 100)
    ks = config.get("ks", [5, 10, 20])
    trees = config.get("ranker_trees", 100)
    if (
        not isinstance(count, int)
        or not 20 <= count <= 2000
        or not isinstance(trees, int)
        or not 1 <= trees <= 500
    ):
        raise ValueError("Invalid candidate count or ranking tree count.")
    if not ks or any(not isinstance(k, int) or not 1 <= k <= count for k in ks):
        raise ValueError("Metric K values must be positive and no larger than candidate count.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = temporal_split(events)
    start = time.perf_counter()
    model, user_ids, item_ids, user_vectors, item_vectors, losses = train_two_tower(split.train, training)
    retrieval = FaissRetriever(item_ids, item_vectors)
    features = FeatureBuilder(split.train, user_ids, item_ids, user_vectors, item_vectors)
    popularity = PopularityRetriever(item_ids, features.popularity)
    train_history = history_sets(split.train)
    rank_groups = candidate_groups(
        split.rank_train, train_history, features.user_map, user_vectors, retrieval, popularity, count
    )
    warm_rank_groups = [group for group in rank_groups if group[0] in features.user_map]
    ranker, rank_training = fit_ranker(warm_rank_groups, features, trees, training.seed)
    report = {
        "schema_version": 1,
        "source": source or {"type": "synthetic", "seed": 42},
        "config": {**asdict(training), "candidates": count, "ranker_trees": trees, "ks": ks},
        "versions": {
            "torch": torch.__version__,
            "faiss": faiss.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "split": {
            "cutoffs": split.cutoffs,
            "rows": {
                name: len(getattr(split, name)) for name in ["train", "rank_train", "validation", "test"]
            },
            "train_users": len(user_ids),
            "train_catalog_items": len(item_ids),
        },
        "training_seconds": time.perf_counter() - start,
        "loss_history": losses,
        "ranking_training": rank_training,
    }
    for phase, earlier in [
        ("validation", [split.train, split.rank_train]),
        ("test", [split.train, split.rank_train, split.validation]),
    ]:
        history = history_sets(pd.concat(earlier))
        groups = candidate_groups(
            getattr(split, phase), history, features.user_map, user_vectors, retrieval, popularity, count
        )
        result, predictions = benchmark(
            groups, history, features.user_map, ranker, features, popularity, ks, count
        )
        report[phase] = result
        (output / f"{phase}_predictions.json").write_text(json.dumps(predictions, indent=2))
    report["limitations"] = [
        "Observed ratings are converted to positive implicit proxies, not real clicks or exposure logs.",
        "Unobserved items are sampled negatives and may be false negatives.",
        "Global chronological splits; no future interactions enter feature statistics or model fitting.",
        "Cold users use training popularity; out-of-catalog targets count as misses.",
        "Evaluation aggregates future-period relevance sets; it is not online next-event replay.",
        "Exact FAISS flat search on a small catalog does not establish million-item ANN scalability.",
        "Fixed hyperparameters; validation is reported without selection or refitting.",
    ]
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame(
        {"feature": features.names, "gain": ranker.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.save(model.state_dict(), artifacts / "two_tower.pt")
    faiss.write_index(retrieval.index, str(artifacts / "items.faiss"))
    ranker.save_model(str(artifacts / "ranker.txt"))
    for name, values in [
        ("user_vectors", user_vectors),
        ("item_vectors", item_vectors),
        ("profiles", features.profiles),
        ("activities", features.activities),
        ("popularity", features.popularity),
        ("recency", features.recency),
    ]:
        np.save(artifacts / f"{name}.npy", values, allow_pickle=False)
    metadata = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "feature_names": features.names,
        "config": report["config"],
        "seen": {str(u): sorted(items) for u, items in history.items()},
        "serving_history_cutoff": split.cutoffs[2],
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    (artifacts / "manifest.json").write_text(
        json.dumps({p.name: sha256(p) for p in artifacts.iterdir() if p.name != "manifest.json"}, indent=2)
    )
    return report
