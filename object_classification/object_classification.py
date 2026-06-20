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
YOLO_WORLD_TOOLS_MODEL_ALIASES = {"yolo-world-tools", "yoloworld-tools", "tools"}
YOLO_WORLD_TOOLS_WEIGHTS = "yolov8s-world.pt"
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
        help="YOLO model weights to use, 'rf-detr-xl', or 'yolo-world-tools' for open-vocabulary tool detection.",
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
    return parser.parse_args()


def read_image_to_buffer(image_path: Path) -> io.BytesIO:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported image type '{image_path.suffix}'. Use one of: {supported}")

    return io.BytesIO(image_path.read_bytes())


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
            label = names[class_id]
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


def rf_detr_label_for_detection(
    class_id: object,
    detection_index: int,
    class_names: object,
    class_name_lookup: list[str] | None,
) -> str:
    if detection_index < len(class_names) and class_names[detection_index]:
        label = str(class_names[detection_index])
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
    if is_yolo_world_tools_model(model_name):
        for class_name in TOOL_CLASS_NAMES:
            print(class_name)
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


def confidence_for_model(args: argparse.Namespace) -> float:
    if args.confidence is not None:
        return args.confidence

    if is_yolo_world_tools_model(args.model):
        return 0.15

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


def run_yolo_detection(image: Any, model_name: str, confidence: float, target_class: str | None = None) -> None:
    configure_model_environment()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics with 'pip install -r requirements.txt'.") from exc

    model = YOLO(model_name)
    results = model.predict(source=image, conf=confidence, verbose=False)
    print_yolo_detections(results, target_class=target_class)


def main() -> None:
    args = parse_args()
    if args.list_classes:
        print_known_classes(args.model)
        return

    if args.image is None:
        raise SystemExit("Image path is required unless --list-classes is used.")

    validate_target_class(args.model, args.target_class)

    image_buffer = read_image_to_buffer(args.image)
    image = load_buffered_image(image_buffer)
    confidence = confidence_for_model(args)

    if is_yolo_world_tools_model(args.model):
        run_yolo_world_tools_detection(image, confidence, target_class=args.target_class)
    elif is_rf_detr_model(args.model):
        run_rf_detr_detection(image, confidence, args.model, target_class=args.target_class)
    else:
        run_yolo_detection(image, args.model, confidence, target_class=args.target_class)


if __name__ == "__main__":
    main()
