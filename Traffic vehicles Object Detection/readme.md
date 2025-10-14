🚗 Vehicle Detection with YOLOv5

This project implements a vehicle detection system using YOLOv5 (You Only Look Once, version 5) for real-time object detection in traffic scenes.
The model is trained to detect multiple vehicle types and license plates with high accuracy.

🧠 Project Overview

This notebook demonstrates a full computer vision pipeline for object detection using YOLOv5, including:

⚙️ Environment setup and dependency installation

🧹 Data preprocessing and validation

🏋️ Model training on a custom dataset

🚘 Detection of vehicles and license plates in images and videos

📂 Dataset

The dataset includes 7 object classes for traffic analysis:

Class	Description
0	Car
1	Number Plate
2	Blur Number Plate
3	Two Wheeler
4	Auto
5	Bus
6	Truck

Dataset structure

Split	Images	Instances
Train	738	—
Validation	185	—
Total	923 images	1,980 object instances

All images are labeled and annotated in YOLO format.

⚙️ Model Training

Configuration

Parameter	Value
Model	yolov5s (small)
Input size	640 × 640 pixels
Epochs	30
Batch size	16
Optimizer	SGD
Learning rate	0.01
📈 Training Progress
Epoch	mAP50	Notes
0	0.049	Initial epoch
5	0.447	Rapid improvement
8	0.557	Consistent accuracy gain

Metrics Tracked

🟩 Box Loss: Bounding-box regression error

🟦 Object Loss: Objectness confidence loss

🟨 Class Loss: Classification error

🎯 Precision (P): True positives / all predictions

🧩 Recall (R): True positives / all actual objects

🏆 mAP50: Mean Average Precision @ IoU 0.5

🧰 Technical Setup

Dependencies

Python 3.11

PyTorch 2.6.0 + CUDA 12.4

Ultralytics YOLOv5

OpenCV

NumPy / Pandas

Hardware

GPU Acceleration via CUDA (for faster training and inference)

🌟 Key Features

✅ Real-time object detection using YOLOv5
✅ Custom 7-class dataset for comprehensive vehicle detection
✅ Automatic data validation to detect missing or corrupt labels
✅ GPU acceleration for efficient model training
✅ Live training progress visualization
✅ Configurable hyperparameters for flexible experimentation

🚀 Usage

Setup environment

git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt


Prepare dataset

Organize images and labels into train/ and val/ directories

Update data.yaml with paths and class names

Train the model

python train.py --img 640 --batch 16 --epochs 30 --data data.yaml --weights yolov5s.pt --device 0


Run inference

python detect.py --source path_to_images_or_video --weights runs/train/exp/weights/best.pt

📊 Applications

This vehicle detection system can be used for:

🚦 Traffic monitoring & analysis
🚘 Automated vehicle counting
📸 License plate recognition
🏙️ Smart city infrastructure
👮 Traffic law enforcement

📈 Performance Summary

The trained YOLOv5 model shows strong detection capabilities across all seven classes, achieving steadily increasing mAP50 through training epochs.
Its lightweight yolov5s architecture provides a balance of speed and accuracy, ideal for real-time vehicle detection tasks.
