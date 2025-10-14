🩺 Breast Cancer Ultrasound Segmentation with U-Net
📘 Project Overview

This project implements a deep learning solution for breast cancer ultrasound image segmentation using the U-Net architecture.
The goal is to automatically segment breast ultrasound images to identify regions of interest (tumor areas) — assisting radiologists in breast cancer detection, diagnosis, and treatment planning.

🧩 Dataset

Source: Breast Ultrasound Images Dataset (Dataset_BUSI_with_GT)

Dataset Composition:

Class	Images	Description
Benign	437	Non-cancerous lesions
Malignant	210	Cancerous lesions
Normal	133	No tumor present

Each image includes a corresponding segmentation mask, with some images having multiple mask annotations that are merged for training.

⚙️ Key Features
🧠 Data Preprocessing

Mask Combination: Merges multiple mask annotations using maximum intensity projection

Resizing: Resizes all images and masks to a consistent resolution

Data Structuring: Creates a well-organized DataFrame containing image paths, mask paths, and class labels

🏗️ Model Architecture

Architecture: Classic U-Net encoder–decoder for semantic segmentation

Skip Connections: Preserve spatial details during upsampling

Framework: Implemented in TensorFlow/Keras

Loss Function: Uses Binary Cross-Entropy or Dice Loss for segmentation accuracy

Data Augmentation: Applied to improve model generalization and robustness

🧱 Project Structure
breast-cancer-ultrasound-unet/
├── data_loading.py             # Dataset loading and preprocessing
├── mask_processing.py          # Mask combination and cleaning
├── model.py                    # U-Net model definition
├── training.py                 # Training loop, loss, and evaluation
├── visualization.py            # Visualization of segmentation results
└── breast-cancer-ultrasound-unet-a0d24c.ipynb  # Main notebook

🧰 Technical Stack
Category	Tools & Libraries
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib
Development	Jupyter Notebook
🩻 Applications

Medical Diagnosis: Assists radiologists in detecting and segmenting tumor regions

Treatment Planning: Supports surgical or biopsy planning

Research: Enables quantitative lesion analysis

Education: Serves as a teaching resource for medical imaging and AI in healthcare

🌍 Potential Impact

This project demonstrates how AI-driven medical imaging can:
✅ Improve diagnostic accuracy
✅ Reduce manual interpretation time
✅ Provide consistent segmentation results
✅ Support early detection of breast cancer

💻 Requirements
pip install tensorflow opencv-python numpy pandas matplotlib jupyter

🚀 Usage

Clone this repository:

git clone https://github.com/yourusername/breast-cancer-ultrasound-unet.git
cd breast-cancer-ultrasound-unet


Open the notebook:

jupyter notebook breast-cancer-ultrasound-unet-a0d24c.ipynb


Run all cells sequentially to:

Load and preprocess data

Train the U-Net model

Visualize segmentation outputs

⚠️ Disclaimer

This is a medical imaging research project intended for educational and experimental purposes only.
Predictions made by the model should not be used for clinical decisions without expert validation.
Always consult qualified medical professionals before acting on model outputs.

🧾 Summary

This project showcases the potential of deep learning in medical imaging, specifically using U-Net for breast ultrasound segmentation.
By combining careful preprocessing, robust architecture, and visualization, it highlights how AI can support—but not replace—human expertise in medical diagnosis.
