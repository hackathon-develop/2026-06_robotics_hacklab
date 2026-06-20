#!/usr/bin/env python3
"""Detect objects in a PNG or JPEG image with a YOLO or RF-DETR model."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
RF_DETR_MODEL_ALIASES = {"rf-detr-xl", "rfdetr-xl", "rfdetr-xlarge", "rf-detr-xlarge"}
RF_DETR_O365_MODEL_ALIASES = {"rf-detr-base-o365", "rfdetr-base-o365"}
YOLO_WORLD_TOOLS_MODEL_ALIASES = {
    "yolo-world-tools",
    "yoloworld-tools",
    "tools",
    "yolov8s-world",
    "yolov8s-world.pt",
    "yolo8s-world",
    "yolo8s-world.pt",
}
YOLO_WORLD_TOOLS_WEIGHTS = "yolov8s-world.pt"
YOLO_MODEL_ALIASES = {
    "yolo11m": "yolo11m.pt",
    "yolo11m.pt": "yolo11m.pt",
}
SAM_MODEL_ALIASES = {
    "sam": "sam_b.pt",
    "sam-b": "sam_b.pt",
    "sam-l": "sam_l.pt",
}
TOOL_CLASS_NAMES = [
    "screwdriver",
    "wrench",
    "pliers",
    "hammer",
    "drill",
    "saw",
    "tape measure",
    "level",
    "utility knife",
    "scissors",
    "bolt",
    "nut",
    "screw",
]
OBJECTS365V2_CLASS_NAMES = [
    "Person",
    "Sneakers",
    "Chair",
    "Other Shoes",
    "Hat",
    "Car",
    "Lamp",
    "Glasses",
    "Bottle",
    "Desk",
    "Cup",
    "Street Lights",
    "Cabinet/shelf",
    "Handbag/Satchel",
    "Bracelet",
    "Plate",
    "Picture/Frame",
    "Helmet",
    "Book",
    "Gloves",
    "Storage box",
    "Boat",
    "Leather Shoes",
    "Flower",
    "Bench",
    "Potted Plant",
    "Bowl/Basin",
    "Flag",
    "Pillow",
    "Boots",
    "Vase",
    "Microphone",
    "Necklace",
    "Ring",
    "SUV",
    "Wine Glass",
    "Belt",
    "Moniter/TV",
    "Backpack",
    "Umbrella",
    "Traffic Light",
    "Speaker",
    "Watch",
    "Tie",
    "Trash bin Can",
    "Slippers",
    "Bicycle",
    "Stool",
    "Barrel/bucket",
    "Van",
    "Couch",
    "Sandals",
    "Bakset",
    "Drum",
    "Pen/Pencil",
    "Bus",
    "Wild Bird",
    "High Heels",
    "Motorcycle",
    "Guitar",
    "Carpet",
    "Cell Phone",
    "Bread",
    "Camera",
    "Canned",
    "Truck",
    "Traffic cone",
    "Cymbal",
    "Lifesaver",
    "Towel",
    "Stuffed Toy",
    "Candle",
    "Sailboat",
    "Laptop",
    "Awning",
    "Bed",
    "Faucet",
    "Tent",
    "Horse",
    "Mirror",
    "Power outlet",
    "Sink",
    "Apple",
    "Air Conditioner",
    "Knife",
    "Hockey Stick",
    "Paddle",
    "Pickup Truck",
    "Fork",
    "Traffic Sign",
    "Ballon",
    "Tripod",
    "Dog",
    "Spoon",
    "Clock",
    "Pot",
    "Cow",
    "Cake",
    "Dinning Table",
    "Sheep",
    "Hanger",
    "Blackboard/Whiteboard",
    "Napkin",
    "Other Fish",
    "Orange/Tangerine",
    "Toiletry",
    "Keyboard",
    "Tomato",
    "Lantern",
    "Machinery Vehicle",
    "Fan",
    "Green Vegetables",
    "Banana",
    "Baseball Glove",
    "Airplane",
    "Mouse",
    "Train",
    "Pumpkin",
    "Soccer",
    "Skiboard",
    "Luggage",
    "Nightstand",
    "Tea pot",
    "Telephone",
    "Trolley",
    "Head Phone",
    "Sports Car",
    "Stop Sign",
    "Dessert",
    "Scooter",
    "Stroller",
    "Crane",
    "Remote",
    "Refrigerator",
    "Oven",
    "Lemon",
    "Duck",
    "Baseball Bat",
    "Surveillance Camera",
    "Cat",
    "Jug",
    "Broccoli",
    "Piano",
    "Pizza",
    "Elephant",
    "Skateboard",
    "Surfboard",
    "Gun",
    "Skating and Skiing shoes",
    "Gas stove",
    "Donut",
    "Bow Tie",
    "Carrot",
    "Toilet",
    "Kite",
    "Strawberry",
    "Other Balls",
    "Shovel",
    "Pepper",
    "Computer Box",
    "Toilet Paper",
    "Cleaning Products",
    "Chopsticks",
    "Microwave",
    "Pigeon",
    "Baseball",
    "Cutting/chopping Board",
    "Coffee Table",
    "Side Table",
    "Scissors",
    "Marker",
    "Pie",
    "Ladder",
    "Snowboard",
    "Cookies",
    "Radiator",
    "Fire Hydrant",
    "Basketball",
    "Zebra",
    "Grape",
    "Giraffe",
    "Potato",
    "Sausage",
    "Tricycle",
    "Violin",
    "Egg",
    "Fire Extinguisher",
    "Candy",
    "Fire Truck",
    "Billards",
    "Converter",
    "Bathtub",
    "Wheelchair",
    "Golf Club",
    "Briefcase",
    "Cucumber",
    "Cigar/Cigarette ",
    "Paint Brush",
    "Pear",
    "Heavy Truck",
    "Hamburger",
    "Extractor",
    "Extention Cord",
    "Tong",
    "Tennis Racket",
    "Folder",
    "American Football",
    "earphone",
    "Mask",
    "Kettle",
    "Tennis",
    "Ship",
    "Swing",
    "Coffee Machine",
    "Slide",
    "Carriage",
    "Onion",
    "Green beans",
    "Projector",
    "Frisbee",
    "Washing Machine/Drying Machine",
    "Chicken",
    "Printer",
    "Watermelon",
    "Saxophone",
    "Tissue",
    "Toothbrush",
    "Ice cream",
    "Hotair ballon",
    "Cello",
    "French Fries",
    "Scale",
    "Trophy",
    "Cabbage",
    "Hot dog",
    "Blender",
    "Peach",
    "Rice",
    "Wallet/Purse",
    "Volleyball",
    "Deer",
    "Goose",
    "Tape",
    "Tablet",
    "Cosmetics",
    "Trumpet",
    "Pineapple",
    "Golf Ball",
    "Ambulance",
    "Parking meter",
    "Mango",
    "Key",
    "Hurdle",
    "Fishing Rod",
    "Medal",
    "Flute",
    "Brush",
    "Penguin",
    "Megaphone",
    "Corn",
    "Lettuce",
    "Garlic",
    "Swan",
    "Helicopter",
    "Green Onion",
    "Sandwich",
    "Nuts",
    "Speed Limit Sign",
    "Induction Cooker",
    "Broom",
    "Trombone",
    "Plum",
    "Rickshaw",
    "Goldfish",
    "Kiwi fruit",
    "Router/modem",
    "Poker Card",
    "Toaster",
    "Shrimp",
    "Sushi",
    "Cheese",
    "Notepaper",
    "Cherry",
    "Pliers",
    "CD",
    "Pasta",
    "Hammer",
    "Cue",
    "Avocado",
    "Hamimelon",
    "Flask",
    "Mushroon",
    "Screwdriver",
    "Soap",
    "Recorder",
    "Bear",
    "Eggplant",
    "Board Eraser",
    "Coconut",
    "Tape Measur/ Ruler",
    "Pig",
    "Showerhead",
    "Globe",
    "Chips",
    "Steak",
    "Crosswalk Sign",
    "Stapler",
    "Campel",
    "Formula 1 ",
    "Pomegranate",
    "Dishwasher",
    "Crab",
    "Hoverboard",
    "Meat ball",
    "Rice Cooker",
    "Tuba",
    "Calculator",
    "Papaya",
    "Antelope",
    "Parrot",
    "Seal",
    "Buttefly",
    "Dumbbell",
    "Donkey",
    "Lion",
    "Urinal",
    "Dolphin",
    "Electric Drill",
    "Hair Dryer",
    "Egg tart",
    "Jellyfish",
    "Treadmill",
    "Lighter",
    "Grapefruit",
    "Game board",
    "Mop",
    "Radish",
    "Baozi",
    "Target",
    "French",
    "Spring Rolls",
    "Monkey",
    "Rabbit",
    "Pencil Case",
    "Yak",
    "Red Cabbage",
    "Binoculars",
    "Asparagus",
    "Barbell",
    "Scallop",
    "Noddles",
    "Comb",
    "Dumpling",
    "Oyster",
    "Table Teniis paddle",
    "Cosmetics Brush/Eyeliner Pencil",
    "Chainsaw",
    "Eraser",
    "Lobster",
    "Durian",
    "Okra",
    "Lipstick",
    "Cosmetics Mirror",
    "Curling",
    "Table Tennis ",
]
COCO_CLASS_NAMES: set[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an image into memory, run object detection, and print labels with positions."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to a PNG or JPEG image.",
    )
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help=(
            "YOLO model weights to use, 'rf-detr-xl', 'yolo-world-tools', "
            "or 'sam'/'sam-b'/'sam-l' for Segment Anything masks."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=(
            "Minimum confidence threshold for detections. Defaults to 0.25 for YOLO, "
            "0.10 for RF-DETR, and 0.15 for yolo-world-tools."
        ),
    )
    parser.add_argument(
        "--target-class",
        help="Optional object class to check for and report. Example: screwdriver.",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Print the known classes for the selected model and exit.",
    )
    parser.add_argument(
        "--sam-box",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="Box prompt for SAM segmentation. Can be repeated.",
    )
    parser.add_argument(
        "--sam-point",
        action="append",
        default=[],
        metavar="X,Y",
        help="Point prompt for SAM segmentation. Can be repeated.",
    )
    parser.add_argument(
        "--sam-label",
        action="append",
        type=int,
        choices=(0, 1),
        default=[],
        help="SAM point label for each --sam-point: 1 foreground, 0 background. Defaults to 1 for all points.",
    )
    parser.add_argument(
        "--sam-name-model",
        default="rf-detr-base-o365",
        help="Model used to name each SAM mask crop. Use yolo-world-tools, rf-detr-base-o365, or a YOLO .pt file.",
    )
    parser.add_argument(
        "--sam-name-confidence",
        type=float,
        default=None,
        help="Minimum confidence threshold for naming SAM mask crops. Defaults to 0.05.",
    )
    parser.add_argument(
        "--no-sam-name",
        action="store_true",
        help="Disable naming SAM masks with a second model.",
    )
    parser.add_argument(
        "--sam-max-name-masks",
        type=int,
        default=10,
        help="Maximum number of SAM masks to pass through the second-stage naming model.",
    )
    return parser.parse_args()


def read_image_to_buffer(image_path: Path) -> io.BytesIO:
    return io.BytesIO(image_path.read_bytes())


def validate_image_path(image_path: Path) -> None:
    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise SystemExit(f"Unsupported image type '{image_path.suffix}'. Use one of: {supported}")


def load_buffered_image(image_buffer: io.BytesIO) -> Any:
    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install Pillow with 'pip install -r requirements.txt'.") from exc

    image_buffer.seek(0)
    try:
        image = Image.open(image_buffer)
        image.verify()
    except UnidentifiedImageError as exc:
        raise ValueError("The input file is not a valid PNG or JPEG image.") from exc

    image_buffer.seek(0)
    return Image.open(image_buffer).convert("RGB")


def print_detection(label: str, confidence: float, xyxy: Iterable[float]) -> None:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    print(
        f"label={label} "
        f"confidence={confidence:.3f} "
        f"position=(x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})"
    )


def print_segmentation(
    label: str,
    confidence: float | None,
    xyxy: Iterable[float],
    area: float,
    mask_confidence: float | None = None,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    confidence_text = f"{confidence:.3f}" if confidence is not None else "n/a"
    message = (
        f"label={label} "
        f"confidence={confidence_text} "
        f"position=(x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}) "
        f"mask_area={area:.0f}"
    )
    if mask_confidence is not None:
        message += f" mask_confidence={mask_confidence:.3f}"

    print(message)


def parse_coordinate_list(raw_values: list[str], expected_length: int, option_name: str) -> list[list[float]]:
    parsed_values = []
    for raw_value in raw_values:
        values = [value.strip() for value in raw_value.split(",")]
        if len(values) != expected_length:
            raise SystemExit(f"{option_name} expects {expected_length} comma-separated numbers: {raw_value}")

        try:
            parsed_values.append([float(value) for value in values])
        except ValueError as exc:
            raise SystemExit(f"{option_name} contains a non-numeric value: {raw_value}") from exc

    return parsed_values


def sam_prompts_from_args(args: argparse.Namespace) -> tuple[list[list[float]] | None, list[list[float]] | None, list[int] | None]:
    bboxes = parse_coordinate_list(args.sam_box, 4, "--sam-box")
    points = parse_coordinate_list(args.sam_point, 2, "--sam-point")

    if args.sam_label and len(args.sam_label) != len(points):
        raise SystemExit("--sam-label must be provided once per --sam-point.")

    labels = args.sam_label or ([1] * len(points))
    return bboxes or None, points or None, labels or None


def mask_bbox_from_data(mask: object) -> list[float]:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(mask, torch.Tensor):
        y_indexes, x_indexes = torch.where(mask > 0)
        if len(x_indexes) == 0 or len(y_indexes) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        return [
            float(x_indexes.min()),
            float(y_indexes.min()),
            float(x_indexes.max()),
            float(y_indexes.max()),
        ]

    return [0.0, 0.0, 0.0, 0.0]


def clamp_xyxy(xyxy: Iterable[float], image: Any) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def mask_to_segment_crop(image: Any, mask: object, xyxy: Iterable[float]) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install Pillow and numpy with 'pip install -r requirements.txt'.") from exc

    mask_array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    mask_array = (mask_array > 0).astype("uint8") * 255
    mask_image = Image.fromarray(mask_array)
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

    masked_image = Image.new("RGB", image.size, (255, 255, 255))
    masked_image.paste(image, mask=mask_image)
    return masked_image.crop(clamp_xyxy(xyxy, image))


def xyxy_to_crop(image: Any, xyxy: Iterable[float]) -> Any:
    return image.crop(clamp_xyxy(xyxy, image))


def resolve_class_label(label: object, class_name_lookup: list[str] | None = None) -> str:
    label_text = str(label)
    class_id = None

    if isinstance(label, int):
        class_id = label
    elif label_text.isdigit():
        class_id = int(label_text)
    elif label_text.startswith("class_") and label_text[6:].isdigit():
        class_id = int(label_text[6:])

    if class_id is not None and class_name_lookup:
        resolved_label = label_from_class_id(class_id, class_name_lookup)
        if resolved_label:
            return resolved_label

    return label_text


def class_name_from_names(names: object, class_id: int, class_name_lookup: list[str] | None = None) -> str:
    label = None
    if isinstance(names, dict):
        label = names.get(class_id, names.get(str(class_id)))
    elif class_id < len(names):
        label = names[class_id]

    if label is None:
        label = f"class_{class_id}"

    return resolve_class_label(label, class_name_lookup=class_name_lookup)


def best_yolo_detection_label(
    results: Iterable[object],
    class_name_lookup: list[str] | None = None,
) -> tuple[str | None, float | None]:
    best_label = None
    best_confidence = None

    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            confidence = float(box.conf[0])
            if best_confidence is None or confidence > best_confidence:
                best_confidence = confidence
                best_label = class_name_from_names(names, int(box.cls[0]), class_name_lookup=class_name_lookup)

    return best_label, best_confidence


def best_rf_detr_detection_label(
    detections: object,
    class_name_lookup: list[str] | None = None,
) -> tuple[str | None, float | None]:
    best_label = None
    best_confidence = None
    class_names = detections.data.get("class_name", [])

    for index, (confidence, class_id) in enumerate(zip(detections.confidence, detections.class_id)):
        confidence = float(confidence)
        if best_confidence is None or confidence > best_confidence:
            best_confidence = confidence
            best_label = rf_detr_label_for_detection(class_id, index, class_names, class_name_lookup)

    return best_label, best_confidence


def load_sam_naming_model(model_name: str, target_class: str | None = None) -> tuple[str, object, list[str] | None]:
    if is_sam_model(model_name):
        raise SystemExit("--sam-name-model must be a naming model, not another SAM model.")

    configure_model_environment()
    if is_yolo_world_tools_model(model_name):
        try:
            from ultralytics import YOLOWorld
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

        model = YOLOWorld(YOLO_WORLD_TOOLS_WEIGHTS)
        model.set_classes(yolo_world_tool_classes(target_class))
        return "yolo", model, None

    if is_rf_detr_model(model_name):
        if is_rf_detr_o365_model(model_name):
            try:
                from rfdetr import RFDETRBase
            except (ImportError, ModuleNotFoundError) as exc:
                raise SystemExit("Missing dependency: install RF-DETR with 'pip install -r requirements.txt'.") from exc

            model = RFDETRBase(pretrain_weights="rf-detr-base-o365.pth", num_classes=365)
            model.model.class_names = ["__background__"] + OBJECTS365V2_CLASS_NAMES
            return "rf-detr", model, OBJECTS365V2_CLASS_NAMES

        try:
            from rfdetr import RFDETRXLarge
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit("Missing dependency: install RF-DETR-XL with 'pip install -r requirements.txt'.") from exc

        return "rf-detr", RFDETRXLarge(num_classes=90), None

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

    return "yolo", YOLO(yolo_weights_for_model(model_name)), None


def name_segment_crop(
    crop: Any,
    naming_model: tuple[str, object, list[str] | None] | None,
    confidence: float,
) -> tuple[str | None, float | None]:
    if naming_model is None:
        return None, None

    model_type, model, class_name_lookup = naming_model
    if model_type == "rf-detr":
        detections = model.predict(crop, threshold=confidence)
        return best_rf_detr_detection_label(detections, class_name_lookup=class_name_lookup)

    results = model.predict(source=crop, conf=confidence, verbose=False)
    return best_yolo_detection_label(results)


def name_segment(
    image: Any,
    mask: object,
    xyxy: Iterable[float],
    naming_model: tuple[str, object, list[str] | None] | None,
    confidence: float,
) -> tuple[str | None, float | None]:
    if naming_model is None:
        return None, None

    crops = [
        mask_to_segment_crop(image, mask, xyxy),
        xyxy_to_crop(image, xyxy),
    ]
    thresholds = [confidence]
    if confidence > 0.01:
        thresholds.append(0.01)

    for threshold in thresholds:
        for crop in crops:
            label, label_confidence = name_segment_crop(crop, naming_model, threshold)
            if label:
                return label, label_confidence

    return None, None


def print_sam_segmentations(
    results: Iterable[object],
    image: Any,
    naming_model: tuple[str, object, list[str] | None] | None = None,
    naming_confidence: float = 0.15,
    max_name_masks: int = 10,
) -> None:
    found_mask = False
    named_mask_count = 0

    for result in results:
        masks = result.masks
        if masks is None or len(masks) == 0:
            continue

        boxes = result.boxes
        box_values = boxes.xyxy if boxes is not None else []
        confidences = boxes.conf if boxes is not None and getattr(boxes, "conf", None) is not None else []

        for index, mask in enumerate(masks.data):
            found_mask = True
            xyxy = box_values[index] if index < len(box_values) else mask_bbox_from_data(mask)
            mask_confidence = float(confidences[index]) if index < len(confidences) else None
            label = f"mask_{index + 1}"
            confidence = mask_confidence
            if naming_model is not None and named_mask_count < max_name_masks:
                named_mask_count += 1
                named_label, named_confidence = name_segment(image, mask, xyxy, naming_model, naming_confidence)
                if named_label:
                    label = named_label
                    confidence = named_confidence

            area = float((mask > 0).sum())
            print_segmentation(label, confidence, xyxy, area, mask_confidence=mask_confidence if naming_model else None)

    if not found_mask:
        print("No masks segmented.")


def print_yolo_detections(results: Iterable[object], target_class: str | None = None) -> None:
    found_detection = False
    found_target = False
    target_class = normalize_label(target_class) if target_class else None

    for result in results:
        names = result.names
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            found_detection = True
            class_id = int(box.cls[0])
            label = class_name_from_names(names, class_id)
            confidence = float(box.conf[0])
            found_target = found_target or normalize_label(label) == target_class
            print_detection(label, confidence, box.xyxy[0])

    if not found_detection:
        print("No objects detected.")
    elif target_class and not found_target:
        print(f"No '{target_class}' detections found.")


def print_rf_detr_detections(
    detections: object,
    target_class: str | None = None,
    class_name_lookup: list[str] | None = None,
) -> None:
    found_detection = False
    found_target = False
    class_names = detections.data.get("class_name", [])
    target_class = normalize_label(target_class) if target_class else None

    for index, (xyxy, confidence, class_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id)
    ):
        found_detection = True
        label = rf_detr_label_for_detection(class_id, index, class_names, class_name_lookup)
        found_target = found_target or normalize_label(label) == target_class
        print_detection(label, float(confidence), xyxy)

    if not found_detection:
        print("No objects detected.")
    elif target_class and not found_target:
        print(f"No '{target_class}' detections found.")


def is_rf_detr_model(model_name: str) -> bool:
    return model_name.strip().lower() in RF_DETR_MODEL_ALIASES | RF_DETR_O365_MODEL_ALIASES


def is_rf_detr_o365_model(model_name: str) -> bool:
    return model_name.strip().lower() in RF_DETR_O365_MODEL_ALIASES


def is_yolo_world_tools_model(model_name: str) -> bool:
    return model_name.strip().lower() in YOLO_WORLD_TOOLS_MODEL_ALIASES


def is_yolo_alias_model(model_name: str) -> bool:
    return model_name.strip().lower() in YOLO_MODEL_ALIASES


def yolo_weights_for_model(model_name: str) -> str:
    return YOLO_MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def is_sam_model(model_name: str) -> bool:
    model_name = model_name.strip().lower()
    return model_name in SAM_MODEL_ALIASES or model_name.endswith(".pt") and "sam" in model_name


def sam_weights_for_model(model_name: str) -> str:
    return SAM_MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def rf_detr_label_for_detection(
    class_id: object,
    detection_index: int,
    class_names: object,
    class_name_lookup: list[str] | None,
) -> str:
    if detection_index < len(class_names) and class_names[detection_index]:
        label = resolve_class_label(class_names[detection_index], class_name_lookup=class_name_lookup)
        if not label.startswith("class_"):
            return label

    class_id = int(class_id)
    if class_name_lookup:
        label = label_from_class_id(class_id, class_name_lookup)
        if label:
            return label

    return f"class_{class_id}"


def label_from_class_id(class_id: int, class_names: list[str]) -> str | None:
    if 1 <= class_id <= len(class_names):
        return class_names[class_id - 1]

    if 0 <= class_id < len(class_names):
        return class_names[class_id]

    return None


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower()


def get_coco_class_names() -> set[str]:
    global COCO_CLASS_NAMES
    if COCO_CLASS_NAMES is None:
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: install RF-DETR with 'pip install -r requirements.txt'.") from exc

        COCO_CLASS_NAMES = {normalize_label(label) for label in COCO_CLASSES.values()}

    return COCO_CLASS_NAMES


def print_known_classes(model_name: str) -> None:
    if is_sam_model(model_name):
        print("SAM segments masks and does not assign object class names.")
        print("Use --sam-box X1,Y1,X2,Y2 or --sam-point X,Y prompts to target specific objects.")
        return

    if is_yolo_world_tools_model(model_name):
        for class_name in TOOL_CLASS_NAMES:
            print(class_name)
        return

    if is_yolo_alias_model(model_name):
        print(f"{model_name} uses the Ultralytics COCO class set.")
        return

    if is_rf_detr_o365_model(model_name):
        for class_id, class_name in enumerate(OBJECTS365V2_CLASS_NAMES, start=1):
            print(f"{class_id}: {class_name}")
        return

    if is_rf_detr_model(model_name):
        for class_name in sorted(get_coco_class_names()):
            print(class_name)
        return

    print("Class listing is only available for RF-DETR and yolo-world-tools models in this script.")


def validate_target_class(model_name: str, target_class: str | None) -> None:
    if not target_class:
        return

    target_class = normalize_label(target_class)
    if is_rf_detr_o365_model(model_name):
        return

    if is_rf_detr_model(model_name) and target_class not in get_coco_class_names():
        raise SystemExit(
            f"'{target_class}' is not in the RF-DETR-XL COCO class list. "
            "Use a fine-tuned screwdriver model or try --model rf-detr-base-o365."
        )


def validate_model_name(model_name: str) -> None:
    if (
        is_sam_model(model_name)
        or is_yolo_world_tools_model(model_name)
        or is_yolo_alias_model(model_name)
        or is_rf_detr_model(model_name)
    ):
        return

    if Path(model_name).is_file():
        return

    if model_name.endswith(".pt"):
        return

    known_models = ", ".join(
        sorted(
            [
                "rf-detr-base-o365",
                "rf-detr-xl",
                "sam",
                "sam-b",
                "sam-l",
                "yolo11m",
                "yolo-world-tools",
                "yolov8s-world",
            ]
        )
    )
    raise SystemExit(
        f"Unknown model '{model_name}'. Use a local .pt file or one of these aliases: {known_models}."
    )


def confidence_for_model(args: argparse.Namespace) -> float:
    if args.confidence is not None:
        return args.confidence

    if is_yolo_world_tools_model(args.model):
        return 0.15

    if is_sam_model(args.model):
        return 0.25

    return 0.10 if is_rf_detr_model(args.model) else 0.25


def configure_model_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


def run_rf_detr_detection(image: Any, confidence: float, model_name: str, target_class: str | None = None) -> None:
    configure_model_environment()

    if is_rf_detr_o365_model(model_name):
        try:
            from rfdetr import RFDETRBase
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit("Missing dependency: install RF-DETR with 'pip install -r requirements.txt'.") from exc

        model = RFDETRBase(pretrain_weights="rf-detr-base-o365.pth", num_classes=365)
        model.model.class_names = ["__background__"] + OBJECTS365V2_CLASS_NAMES
    else:
        try:
            from rfdetr import RFDETRXLarge
        except (ImportError, ModuleNotFoundError) as exc:
            raise SystemExit("Missing dependency: install RF-DETR-XL with 'pip install -r requirements.txt'.") from exc

        model = RFDETRXLarge(num_classes=90)

    detections = model.predict(image, threshold=confidence)
    class_name_lookup = OBJECTS365V2_CLASS_NAMES if is_rf_detr_o365_model(model_name) else None
    print_rf_detr_detections(detections, target_class=target_class, class_name_lookup=class_name_lookup)


def yolo_world_tool_classes(target_class: str | None = None) -> list[str]:
    if target_class:
        return [normalize_label(target_class)]

    return TOOL_CLASS_NAMES


def run_yolo_world_tools_detection(image: Any, confidence: float, target_class: str | None = None) -> None:
    configure_model_environment()

    try:
        from ultralytics import YOLOWorld
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

    model = YOLOWorld(YOLO_WORLD_TOOLS_WEIGHTS)
    model.set_classes(yolo_world_tool_classes(target_class))
    results = model.predict(source=image, conf=confidence, verbose=False)
    print_yolo_detections(results, target_class=target_class)


def run_sam_segmentation(
    image: Any,
    confidence: float,
    model_name: str,
    bboxes: list[list[float]] | None,
    points: list[list[float]] | None,
    labels: list[int] | None,
    name_model: str | None,
    target_class: str | None = None,
    name_confidence: float | None = None,
    max_name_masks: int = 10,
) -> None:
    configure_model_environment()

    try:
        from ultralytics import SAM
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

    model = SAM(sam_weights_for_model(model_name))
    results = model.predict(
        source=image,
        bboxes=bboxes,
        points=points,
        labels=labels,
        conf=confidence,
        verbose=False,
    )
    naming_model = load_sam_naming_model(name_model, target_class=target_class) if name_model else None
    naming_confidence = name_confidence if name_confidence is not None else 0.05
    print_sam_segmentations(
        results,
        image,
        naming_model=naming_model,
        naming_confidence=naming_confidence,
        max_name_masks=max_name_masks,
    )


def run_yolo_detection(image: Any, model_name: str, confidence: float, target_class: str | None = None) -> None:
    configure_model_environment()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

    model = YOLO(yolo_weights_for_model(model_name))
    results = model.predict(source=image, conf=confidence, verbose=False)
    print_yolo_detections(results, target_class=target_class)


def main() -> None:
    args = parse_args()
    if args.list_classes:
        print_known_classes(args.model)
        return

    if args.image is None:
        raise SystemExit("Image path is required unless --list-classes is used.")

    if args.sam_max_name_masks < 0:
        raise SystemExit("--sam-max-name-masks must be 0 or greater.")

    validate_image_path(args.image)
    validate_model_name(args.model)
    validate_target_class(args.model, args.target_class)

    image_buffer = read_image_to_buffer(args.image)
    image = load_buffered_image(image_buffer)
    confidence = confidence_for_model(args)

    if is_sam_model(args.model):
        bboxes, points, labels = sam_prompts_from_args(args)
        name_model = None if args.no_sam_name else args.sam_name_model
        run_sam_segmentation(
            image,
            confidence,
            args.model,
            bboxes,
            points,
            labels,
            name_model,
            target_class=args.target_class,
            name_confidence=args.sam_name_confidence,
            max_name_masks=args.sam_max_name_masks,
        )
    elif is_yolo_world_tools_model(args.model):
        run_yolo_world_tools_detection(image, confidence, target_class=args.target_class)
    elif is_rf_detr_model(args.model):
        run_rf_detr_detection(image, confidence, args.model, target_class=args.target_class)
    else:
        run_yolo_detection(image, args.model, confidence, target_class=args.target_class)


if __name__ == "__main__":
    main()
