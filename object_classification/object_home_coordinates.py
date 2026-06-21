#!/usr/bin/env python3
"""Hardcoded home coordinates for object classes.

The coordinate frame is centered at the workspace midpoint: (0, 0).
Positive X is to the right and positive Y is forward/up in the workspace.
Adjust the HOME_COORDINATES values after deciding the final home positions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HomeCoordinate:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


# Hardcoded home coordinates for the four supported classes.
# Replace the class names/values here with your final object classes and homes.
HOME_COORDINATES: dict[str, HomeCoordinate] = {
    "marker": HomeCoordinate(x=-0.17, y=0.12, z=0.0, yaw=0.0),
    "canned": HomeCoordinate(x=0.18, y=0.10, z=0.0, yaw=0.0),
    "spoon": HomeCoordinate(x=-0.18, y=-0.12, z=0.0, yaw=0.0),
    "mouse": HomeCoordinate(x=0.18, y=-0.12, z=0.0, yaw=0.0),
}


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace("_", " ")


def home_coordinate_for_class(class_name: str) -> HomeCoordinate:
    normalized_name = normalize_class_name(class_name)
    try:
        return HOME_COORDINATES[normalized_name]
    except KeyError as exc:
        known_classes = ", ".join(sorted(HOME_COORDINATES))
        raise KeyError(f"Unknown class '{class_name}'. Known classes: {known_classes}") from exc


def home_coordinates_as_dict() -> dict[str, dict[str, float]]:
    return {
        class_name: asdict(coordinate)
        for class_name, coordinate in HOME_COORDINATES.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return hardcoded home coordinates for object classes."
    )
    parser.add_argument(
        "class_name",
        nargs="?",
        help="Object class to look up. Omit to print every home coordinate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.class_name:
        coordinate = home_coordinate_for_class(args.class_name)
        print(json.dumps({normalize_class_name(args.class_name): asdict(coordinate)}, indent=2))
        return

    print(json.dumps(home_coordinates_as_dict(), indent=2))


if __name__ == "__main__":
    main()
