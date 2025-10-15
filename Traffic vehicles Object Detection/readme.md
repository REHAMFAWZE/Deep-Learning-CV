# 🚀 Real-Time Traffic Object Detection using YOLOv8

This project focuses on **real-time traffic object detection** using the **YOLOv8** model.  
The model detects and classifies multiple traffic-related objects such as **cars, number plates, two-wheelers, buses, trucks, autos**, and even **blurred number plates** — showcasing the power of modern **computer vision** for **intelligent transportation systems**.

---

## 🔍 Project Highlights

- **Dataset:** Custom traffic dataset with **7 object classes**  
- **Model:** YOLOv8m (fine-tuned for **30 epochs**)  
- **Performance:**  
  - **mAP@50:** 0.822  
  - **mAP@50-95:** 0.583  
- **Inference:** Real-time detection on validation images with bounding boxes and class labels  

---

## 📊 Key Results

| Object Class           | mAP@50 |
|-------------------------|--------|
| 🚗 Car                 | **0.950** |
| 🔢 Number Plate        | **0.855** |
| 🏍️ Two-Wheeler         | **0.919** |
| 🚌 Bus                 | **0.840** |
| 🚚 Truck               | **0.773** |
| 🚖 Auto / Three-Wheeler| — |
| 🌀 Blurred Plate       | — |

*(Add the last two values when available.)*

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Model:** Ultralytics YOLOv8  
- **Image Processing:** OpenCV  
- **Training Environment:** Kaggle GPU (accelerated training)

---

## ⚙️ Workflow Overview

1. **Dataset Preparation**
   - Custom dataset with labeled bounding boxes.
   - Preprocessed and split into training and validation sets.

2. **Model Training**
   - Fine-tuned YOLOv8m on custom dataset for 30 epochs.
   - Used hyperparameter tuning for optimal learning rate and augmentation.

3. **Evaluation**
   - Calculated mAP@50 and mAP@50-95.
   - Visualized detection results on validation images.

4. **Inference**
   - Performed real-time detection on sample images and video streams.

---

## 📈 Next Steps

- ✅ **Deploy** the model for real-time **video inference**
- ⚙️ **Optimize** for **edge devices**
- 🔧 Apply **model pruning** and **quantization** for faster inference
- 🌐 Integrate into a **traffic monitoring system dashboard**

---

