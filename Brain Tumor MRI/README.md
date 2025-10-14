📁 Project Overview
This Jupyter notebook implements a multi-model classification pipeline for detecting brain tumors from MRI scans. The dataset contains labeled MRI images categorized into four classes:

Glioma

Meningioma

Pituitary

No Tumor

The project explores and compares the performance of several deep learning models, including custom CNNs and pre-trained architectures like VGG16 and ResNet50, to classify brain tumor types accurately.

🧠 Objectives
Perform data exploration and visualization of the brain tumor MRI dataset.

Implement data preprocessing and augmentation techniques.

Build and train multiple deep learning models for classification.

Evaluate model performance using confusion matrices, classification reports, and accuracy metrics.

Compare results across different architectures.

📊 Dataset
Source: Kaggle — Brain Tumor MRI Dataset

Training Samples: 5,712

Testing Samples: 1,311

Classes: 4 (Glioma, Meningioma, Pituitary, No Tumor)

Image Format: .jpg

The dataset is well-balanced, reducing the risk of model bias and enabling reliable performance across all classes.

🛠️ Models Implemented
Custom CNN:

Convolutional layers with pooling and dropout

Fully connected layers for classification

Transfer Learning Models:

VGG16 (with fine-tuning)

ResNet50 (with custom top layers)

Training Features:

Data augmentation (rotation, zoom, flip, etc.)

Early stopping to prevent overfitting

Optimizers: Adam, SGD

📈 Evaluation Metrics
Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Classification Report


📁 Notebook Structure
Environment Setup & Imports

Data Loading & Exploration

Data Preprocessing & Augmentation

Model Building:

Custom CNN

VGG16-based model

ResNet50-based model

Model Training & Evaluation

Performance Comparison & Visualization

Report Generation (using reportlab)

🚀 How to Run
Prerequisites
Python 3.7+

TensorFlow 2.x

Jupyter Notebook

Libraries: pandas, matplotlib, seaborn, scikit-learn, PIL, reportlab

Steps
Upload the notebook to your Kaggle or local environment.

Ensure the dataset is available at the path:

text
/kaggle/input/brain-tumor-mri-dataset/
Run the cells sequentially.

Modify hyperparameters or model architectures as needed.

📌 Key Features
Modular Code: Functions for data loading, visualization, and model training.

Data Augmentation: Improves model generalization.

Transfer Learning: Leverages pre-trained models for better accuracy.

Visualization: Sample images per class, training/validation curves, confusion matrices.

PDF Reporting: Automated generation of training summaries.

📎 Example Outputs
Sample images from each tumor class

Training/validation accuracy and loss plots

Confusion matrices and classification reports

Performance comparison across models

🧪 Possible Improvements
Experiment with other pre-trained models (e.g., Inception, EfficientNet)

Implement class-weighted loss if class imbalance arises

Use gradient accumulation for larger batch sizes

Try ensemble methods for improved performance

👨‍💻 Author
This notebook was developed as part of a deep learning project for medical image classification. Contributions and feedback are welcome.

📜 License
This project is for educational and research purposes. Please ensure proper attribution when using or modifying the code.

Dataset link:https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Happy Coding! 🧠🔬


