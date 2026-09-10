#!/usr/bin/env python3
"""Build the drift reference using the original deterministic TRAIN split only."""

import json

from app.config import Settings
from app.model_service import artifact_sha256
from app.monitoring import build_profile

from churnguard import config, data


def main():
    training = data.make_splits(data.prepare_dataset(data.load_raw_data())).X_train
    profile = build_profile(training)
    profile["source_dataset_sha256"] = artifact_sha256(config.RAW_DATA_PATH.read_bytes())
    path = Settings().reference_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    print(f"Built training reference: {len(training)} rows")


if __name__ == "__main__":
    main()
