"""CLI entrypoint for feature extraction."""

import argparse

from neuromm26_baseline.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feature_extract.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Feature extraction placeholder with config: {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
