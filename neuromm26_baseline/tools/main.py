"""Unified CLI dispatcher for the baseline package."""

from __future__ import annotations

import argparse
import sys

from . import eval as eval_tool
from . import infer as infer_tool
from . import train as train_tool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval", "infer"])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.command == "train":
        return train_tool.main()
    if args.command == "eval":
        return eval_tool.main()
    return infer_tool.main()


if __name__ == "__main__":
    raise SystemExit(main())
