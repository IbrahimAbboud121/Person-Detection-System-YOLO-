# Person Detection System (YOLOv8)

A real-time person detection system built with YOLOv8 and OpenCV, featuring
a custom-trained model using Roboflow for dataset preparation and Google Colab
for training.

## How it works

A custom YOLOv8 model was trained on a labeled person dataset and deployed
for real-time inference via webcam. Each frame is passed through the model,
which draws bounding boxes around detected persons with confidence scores.

## Pipeline
```
Webcam ──► YOLOv8 Inference ──► Bounding Box Annotation ──► Display
```

### Training
- Dataset labeled and exported using **Roboflow**
- Model trained on **Google Colab** (GPU)
- Base model: YOLOv8 (Ultralytics)

### Inference
- Real-time webcam feed via OpenCV
- Frame flipped horizontally for natural mirror view
- Annotated frames rendered using Ultralytics built-in `.plot()`

## Run it
```bash
pip install ultralytics opencv-python
python detect.py
```

> Update the model path in `detect.py` to point to your `.pt` file.

## Skills demonstrated

- Custom YOLO model training on a self-prepared dataset
- Dataset collection and annotation with Roboflow
- Real-time inference pipeline with OpenCV
- GPU-accelerated training on Google Colab
