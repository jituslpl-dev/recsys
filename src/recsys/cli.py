from pathlib import Path
import argparse
import json
import pandas as pd
from recsys.data import synthetic_events, sha256
from recsys.pipeline import run
from recsys.service import RecommendationService


def main():
    parser = argparse.ArgumentParser(description="Two-stage recommendation research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--input", help="CSV with user_id,item_id,timestamp")
    training.add_argument("--config", default="config/default.json")
    training.add_argument("--output", default="reports/synthetic")
    serving = commands.add_parser("recommend")
    serving.add_argument("--artifacts", required=True)
    serving.add_argument("--user", required=True, type=int)
    serving.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.command == "recommend":
        print(json.dumps(RecommendationService(args.artifacts).recommend(args.user, args.k), indent=2))
        return
    config = json.loads(Path(args.config).read_text())
    if args.input:
        events = pd.read_csv(args.input)
        source = {"type": "CSV", "input_sha256": sha256(args.input)}
        metadata = Path(args.input).parent / "source.json"
        if metadata.exists():
            source["metadata"] = json.loads(metadata.read_text())
    else:
        events = synthetic_events(seed=config.get("seed", 42))
        source = {"type": "synthetic", "seed": config.get("seed", 42)}
    report = run(events, args.output, config, source)
    print(json.dumps({"output": args.output, "split": report["split"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
