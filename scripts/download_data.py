#!/usr/bin/env python3
"""Fetch the documented upstream data revision and verify the frozen dataset checksum."""

import hashlib
from urllib.request import urlopen

from churnguard.config import RAW_DATA_PATH

SOURCE_REVISION = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{SOURCE_REVISION}/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def main():
    if RAW_DATA_PATH.exists():
        content = RAW_DATA_PATH.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch; existing files were not overwritten")
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.write_bytes(content)
    print("Verified frozen IBM Telco dataset")


if __name__ == "__main__":
    main()
