import numpy as np


def user_metrics(recommendations, relevant, k):
    if not isinstance(k, int) or k < 1 or not relevant:
        raise ValueError("Positive integer K and nonempty relevant set required.")
    recommendations = list(recommendations)
    if len(set(recommendations)) != len(recommendations):
        raise ValueError("Duplicate recommendation identifiers are invalid.")
    relevant = set(relevant)
    hits = np.asarray([item in relevant for item in recommendations[:k]], dtype=float)
    ranks = np.flatnonzero(hits) + 1
    dcg = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)))
    ideal = float(np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2)))
    return {
        "recall": float(hits.sum() / len(relevant)),
        "ndcg": dcg / ideal,
        "mrr": 0.0 if len(ranks) == 0 else float(1 / ranks[0]),
        "hit_rate": float(bool(len(ranks))),
    }


def evaluate(predictions, relevance, ks=(5, 10, 20)):
    if not relevance:
        raise ValueError("Evaluation needs at least one eligible user.")
    result = {"users": len(relevance), "metrics": {}}
    for k in ks:
        per_user = [
            user_metrics(predictions.get(user, []), targets, k) for user, targets in relevance.items()
        ]
        result["metrics"][f"@{k}"] = {
            name: float(np.mean([row[name] for row in per_user])) for name in per_user[0]
        }
    return result
