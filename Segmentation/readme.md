🩺 Breast Ultrasound Image Segmentation & Classification
📘 Project Overview

This Jupyter Notebook implements a deep learning pipeline for analyzing breast ultrasound images, performing both segmentation (to identify tumor regions) and classification (to categorize images as benign, malignant, or normal).
The project demonstrates a foundational approach to medical image analysis using deep learning techniques.

🧩 Dataset

Source: Breast Ultrasound Images Dataset (Dataset_BUSI_with_GT)

Image Size: 256×256 pixels (grayscale)
Total Samples: 780 images with corresponding ground truth masks

📊 Class Distribution
Class	Percentage	Description
Benign	~57.2%	Non-cancerous lesions
Malignant	~17.1%	Cancerous lesions
Normal	~25.7%	No tumor present

⚠️ Note: The dataset is imbalanced, with a higher number of benign cases.

⚙️ Key Features
🧠 Data Loading & Preprocessing

Loads both ultrasound images and segmentation masks

Handles missing masks by generating zero-filled arrays

Normalizes pixel values to [0, 1]

Resizes all images to 256×256 resolution

🔍 Data Analysis

Class distribution visualization (bar & pie charts)

Random sample visualization with corresponding labels

Dataset shuffling and stratified splitting

🧾 Data Splitting
Subset	Samples	Percentage
Training	546	70%
Validation	117	15%
Test	117	15%

Maintains class proportions using stratified sampling.

🎨 Visualization

display_random_image() function to preview random images

Side-by-side display of ultrasound images and segmentation masks

🧠 Technical Stack
Purpose	Libraries Used
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Progress Monitoring	tqdm
🏗️ Model Architecture (Next Steps)

The notebook establishes the foundation for a dual-task model integrating:

Segmentation Network — U-Net or similar CNN-based architecture to segment tumor regions.

Classification Network — CNN-based model for tumor classification (benign, malignant, normal).

Multi-task Learning Setup — Joint training for segmentation and classification for improved diagnostic accuracy.

🧪 Potential Applications

Automated breast cancer detection and diagnosis

Tumor localization and segmentation in ultrasound images

Computer-aided diagnostic (CAD) systems for radiologists

Research in medical image analysis and multi-task learning

🧰 Requirements
pip install tensorflow opencv-python numpy pandas matplotlib seaborn tqdm

🚀 Usage

Clone the repository:

git clone https://github.com/yourusername/breast-ultrasound-segmentation.git


Open the Jupyter Notebook in Colab or Jupyter Lab.

Run all cells sequentially to:

Load and preprocess the dataset

Visualize sample images and masks

Prepare data for segmentation and classification models

📈 Future Work

Implement U-Net for tumor segmentation

Develop CNN classifier for tumor type prediction

Introduce data augmentation to reduce class imbalance

Evaluate models using Dice coefficient, IoU, accuracy, and F1-score

Visualize segmentation and classification results

💡 Summary

This project demonstrates the first stages of building a medical image analysis pipeline using deep learning.
It combines segmentation and classification tasks to support early breast cancer detection, contributing toward AI-driven diagnostic tools in healthcare.
