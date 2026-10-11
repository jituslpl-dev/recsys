"""Fetch the official stable MovieLens 100K research dataset; raw data is Git-ignored."""

from pathlib import Path
from io import BytesIO
import hashlib
import json
import urllib.request
import zipfile
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Fetch and convert MovieLens 100K")
    parser.add_argument("--archive", help="Previously downloaded official ml-100k.zip")
    args = parser.parse_args()
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "RecSys-research-project/1.0"})
    if args.archive:
        raw = Path(args.archive).read_bytes()
    else:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Unexpected dataset archive size.")
    output = Path("data/movielens")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        ratings = pd.read_csv(
            BytesIO(archive.read("ml-100k/u.data")),
            sep="\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        (output / "DATASET_README.txt").write_bytes(archive.read("ml-100k/README"))
    # Every observed rating event is a positive consumption proxy; explicit score is discarded.
    events = ratings[["user_id", "item_id", "timestamp"]]
    events.to_csv(output / "events.csv", index=False)
    source = {
        "dataset": "MovieLens 100K",
        "publisher": "GroupLens Research, University of Minnesota",
        "url": url,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "events_sha256": hashlib.sha256((output / "events.csv").read_bytes()).hexdigest(),
        "rows": len(events),
        "users": int(events.user_id.nunique()),
        "items": int(events.item_id.nunique()),
        "implicit_conversion": "Observed rating is positive consumption proxy; numeric rating discarded.",
        "license": "Research use under GroupLens dataset terms; read DATASET_README.txt. No endorsement implied.",
    }
    (output / "source.json").write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
