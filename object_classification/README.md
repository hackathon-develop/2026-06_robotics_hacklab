# Object Classification

Command-line object detection for PNG and JPEG images using a YOLO model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python object_classification.py path/to/image.jpg
```

Use a specific YOLO weights file:

```bash
python object_classification.py path/to/image.png --model path/to/model.pt
```

The script reads the image into an in-memory buffer, loads it from that buffer, runs YOLO detection, and prints each detected object label with its bounding-box position.
