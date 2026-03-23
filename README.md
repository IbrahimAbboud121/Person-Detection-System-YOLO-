# Person Detection System (YOLOv8)
youtube:https://youtu.be/c3d2QZIKrUY
A real-time person detection system built with a fully custom-trained YOLOv8
model — from raw image collection to live webcam inference.

## How it works

Every part of this project was built from scratch: images were collected
manually, labeled using Roboflow, trained on Google Colab, and deployed
for real-time inference via webcam.

## Full Pipeline
```
Image Collection ──► Labeling (Roboflow) ──► Training (Colab) ──► Real-time Inference (OpenCV)
```

### Dataset
- Images collected **manually** (real-world photos)
- Annotated and exported using **Roboflow**
- Custom single-class dataset: `person`

### Training
- Base model: YOLOv8 (Ultralytics)
- Trained on **Google Colab** with GPU acceleration
- Output: custom `.pt` weights file

### Inference
- Real-time webcam feed via OpenCV
- Frame flipped horizontally for natural mirror view
- Bounding boxes and confidence scores rendered per frame

## Run it
```bash
pip install ultralytics opencv-python
python detect.py
```

> Update the model path in `detect.py` to point to your `.pt` file.

## Skills demonstrated

- End-to-end custom object detection pipeline
- Manual dataset collection and real-world image gathering
- Dataset annotation and management with Roboflow
- YOLOv8 fine-tuning on a custom dataset
- Real-time inference with OpenCV and Ultralytics
