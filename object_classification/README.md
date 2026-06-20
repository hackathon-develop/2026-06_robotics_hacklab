# Object Classification

Command-line object detection from a camera feed using a YOLO model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python object_classification.py 0
```

Use a specific YOLO weights file:

```bash
python object_classification.py 0 --model path/to/model.pt
```

Filter to a comma-separated list of target classes:

```bash
python object_classification.py 0 --model rf-detr-base-o365 --target-classes screwdriver,hammer,wrench
```

The script opens the camera ID, runs detection in a loop, prints each detected object label with its bounding-box position, and displays an annotated camera window. Press `q` in the camera window or `Ctrl+C` to stop.
