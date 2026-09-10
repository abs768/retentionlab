#!/usr/bin/env python3
"""Check trusted artifact integrity, feature exclusion, class ordering and smoke inference."""

from app.config import Settings
from app.model_service import ArtifactError, ModelService


def main() -> int:
    try:
        service = ModelService.load(Settings())
    except ArtifactError as exc:
        print(f"Artifact verification failed: {exc}")
        return 1
    print(f"Verified {service.metadata['model_version']}: {service.metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
