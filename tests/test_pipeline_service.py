import json
import numpy as np
import pytest
from recsys.data import synthetic_events
from recsys.pipeline import run, candidate_groups
from recsys.service import RecommendationService
from recsys.retrieval import FaissRetriever, PopularityRetriever


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    path = tmp_path_factory.mktemp("experiment")
    report = run(
        synthetic_events(users=30, items=80, interactions=20),
        path,
        {
            "epochs": 3,
            "embedding_dim": 8,
            "negatives": 2,
            "candidates": 30,
            "ranker_trees": 10,
            "ks": [5, 10, 20],
        },
    )
    return path, report


def test_end_to_end_groups_reports_and_candidate_ceiling(experiment):
    path, report = experiment
    assert report["ranking_training"]["usable_query_groups"] >= 5
    assert report["ranking_training"]["forced_positive_injection"] is False
    assert (
        report["loss_history"][-1]["pairwise_logistic_loss"]
        < report["loss_history"][0]["pairwise_logistic_loss"]
    )
    for phase in ["validation", "test"]:
        candidates = report[phase]["candidate_recall"]["metrics"]["@30"]["recall"]
        assert report[phase]["two_stage"]["metrics"]["@20"]["recall"] <= candidates + 1e-12
        predictions = json.loads((path / f"{phase}_predictions.json").read_text())
        assert all(set(p["ranked"]) == set(p["retrieved"]) for p in predictions)


def test_saved_service_matches_test_predictions_and_excludes_seen(experiment):
    path, _ = experiment
    service = RecommendationService(path / "artifacts")
    predictions = json.loads((path / "test_predictions.json").read_text())
    for row in predictions:
        output = service.recommend(row["user_id"], 10)
        assert [r["item_id"] for r in output["recommendations"]] == row["ranked"][:10]
        assert not set(row["ranked"]) & service.seen.get(row["user_id"], set())
    assert service.recommend(999999)["strategy"] == "cold_user_popularity"
    with pytest.raises(ValueError):
        service.recommend(0, k=1000)


def test_integrity_failure_before_binary_load(experiment, tmp_path):
    import shutil

    path, _ = experiment
    shutil.copytree(path / "artifacts", tmp_path / "broken")
    (tmp_path / "broken" / "ranker.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="integrity"):
        RecommendationService(tmp_path / "broken")


def test_targets_are_not_forced_into_retrieval():
    import pandas as pd

    vectors = np.eye(3, dtype=np.float32)
    retrieval = FaissRetriever([1, 2, 3], vectors)
    groups = candidate_groups(
        pd.DataFrame({"user_id": [0], "item_id": [3]}),
        {},
        {0: 0},
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        retrieval,
        PopularityRetriever([1, 2, 3], [3, 2, 1]),
        1,
    )
    assert groups[0][1] == [(1, 1.0)]
    assert groups[0][2] == {3}
