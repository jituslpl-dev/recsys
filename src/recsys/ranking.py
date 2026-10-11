import numpy as np
import lightgbm as lgb


def fit_ranker(groups, features, trees=100, seed=42):
    matrices, labels, sizes = [], [], []
    usable = 0
    for user, candidates, relevant in groups:
        y = np.array([int(item in relevant) for item, _ in candidates])
        # Do not inject targets into candidates: train only on actual retrieval outputs.
        if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
            continue
        matrices.append(features.build(user, candidates))
        labels.append(y)
        sizes.append(len(y))
        usable += 1
    if usable < 5:
        raise ValueError(
            "At least five retrieved query groups with positive and negative candidates required."
        )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=trees,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(np.vstack(matrices), np.concatenate(labels), group=sizes, feature_name=features.names)
    return model.booster_, {
        "usable_query_groups": usable,
        "candidate_rows": sum(sizes),
        "discarded_uninformative_groups": len(groups) - usable,
        "positive_candidates": int(np.concatenate(labels).sum()),
        "forced_positive_injection": False,
    }


def rerank(booster, features, user, candidates):
    if not candidates:
        return []
    scores = booster.predict(features.build(user, candidates), num_threads=2)
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index][0]))
    return [(int(candidates[index][0]), float(scores[index])) for index in order]
