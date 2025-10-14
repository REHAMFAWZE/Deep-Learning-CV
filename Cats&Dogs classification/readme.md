🐾 Cats vs Dogs Image Classification

This project implements a deep learning model to classify images of cats and dogs using the Microsoft Cats vs Dogs dataset.
It demonstrates a full computer vision pipeline — from data loading and preprocessing to exploratory analysis and model preparation — for binary image classification.

🧠 Overview

The notebook provides a structured workflow for developing an image classifier, covering:

📦 Data loading and inspection

🔍 Exploratory data analysis (EDA)

🧹 Image preprocessing

🧱 Model-ready dataset preparation

The goal is to build and prepare a dataset suitable for training a convolutional neural network (CNN) to accurately distinguish between cats and dogs.

📂 Dataset Details

Source: Microsoft Cats vs Dogs Dataset

Location: /kaggle/input/microsoft-catsvsdogs-dataset/PetImages

Class	Label	Approx. Count
🐶 Dog	0	~12,500
🐱 Cat	1	~12,498

Total Images: ~24,998
Image Dimensions: 150 × 150 × 3 (RGB)

⚙️ Project Structure
1️⃣ Environment Setup

Import essential libraries: TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, and PIL.

2️⃣ Data Loading

Read images from directory structure

Create class labels using a dictionary mapping (e.g., {'Cat': 1, 'Dog': 0})

Automatically skip corrupted or unreadable image files

3️⃣ Data Analysis

Check class balance (dataset is 50% cats / 50% dogs)

Visualize class distribution using bar and pie charts

Display random samples with labels for visual inspection

4️⃣ Image Preprocessing

Resize all images to 150×150 pixels

Convert from BGR → RGB format

Normalize pixel values to [0,1] range

5️⃣ Data Visualization

Random image grid to confirm correct preprocessing and labeling

Visual inspection to detect noise or quality issues

6️⃣ Model Preparation

Organize preprocessed data into arrays

Prepare training and validation sets for model input

🧩 Key Features

✅ Balanced dataset — equal number of cat and dog images
✅ Automated error handling for corrupted images
✅ Clean preprocessing pipeline — resize, normalize, encode
✅ Visual insights — class balance, sample displays
✅ Ready for CNN training

🧰 Requirements

Install all required dependencies using:

pip install tensorflow numpy pandas matplotlib seaborn pillow

🚀 Usage

Download or import the dataset from Kaggle.

Open the notebook in Jupyter Notebook or Google Colab.

Run cells sequentially to:

Load and preprocess data

Explore and visualize the dataset

Prepare the data for CNN model training

⚠️ Note

Some images in the dataset are corrupted or incomplete.
These are automatically skipped with informative warnings during data loading — ensuring the final dataset is clean and consistent for model training.

🧾 Summary

This project sets up a solid foundation for binary image classification using deep learning.
It provides a clean preprocessing pipeline and ensures dataset quality, making it ideal for training CNN models such as VGG16, ResNet, or custom architectures.
