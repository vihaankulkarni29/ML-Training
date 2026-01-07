"""Entry point for inhibition zone detection demo."""
from __future__ import annotations

import argparse
from pathlib import Path

from detect_zones import detect_zones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect inhibition zones from plate image")
    parser.add_argument("--image", type=Path, default=Path("data/test_plate.jpg"), help="Path to plate image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zones = detect_zones(args.image)
    if zones:
        print(f"Detected {len(zones)} zones: {zones}")
    else:
        print("No zones detected (stub implementation)")


if __name__ == "__main__":
    main()
