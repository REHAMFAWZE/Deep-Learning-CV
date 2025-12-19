Project Title:

Chest X-Ray Classification using Convolutional Neural Networks (CNNs) 🩺

1️⃣ Problem Statement

Pneumonia is a serious lung infection that can be life-threatening if not detected early. Chest X-ray imaging is a standard method for diagnosis, but manual analysis is time-consuming and prone to errors.

Goal: Develop a deep learning model that automatically classifies chest X-ray images into:

NORMAL (healthy lungs)

PNEUMONIA (infected lungs)

This automation can assist doctors in faster and more accurate diagnosis.

2️⃣ Dataset

Source: Chest X-Ray dataset (Kaggle)

Train / Validation / Test split:

train/ → used to train the CNN

val/ → used for model validation

test/ → used for final evaluation

Classes:

NORMAL

PNEUMONIA

Challenge: The dataset was imbalanced:

25% NORMAL

75% PNEUMONIA

Handling this imbalance was crucial to avoid biased predictions.

3️⃣ Data Preprocessing

Images resized to 224x224 pixels for consistency.

Converted to RGB if necessary.

Normalized pixel values to [0,1].

Visualized sample images to confirm loading.

4️⃣ Data Augmentation & Class Balancing

Augmentation techniques:

Rotation, width/height shifts, shear, zoom, horizontal flips

Helps increase training diversity and reduce overfitting

Class weights:

Computed to give more importance to the minority class (NORMAL) during training

Ensures the model doesn’t simply predict the majority class

5️⃣ Model Architecture (CNN)

Input: (224, 224, 3) images

3 Convolutional Blocks:

Conv2D → Conv2D → MaxPooling → Dropout

Conv2D → Conv2D → MaxPooling → Dropout

Conv2D → Conv2D → MaxPooling → Dropout

Flatten layer

Dense layers with Dropout: 256 → 128 neurons

Output: 1 neuron with sigmoid activation (binary classification)

Optimizer & Loss:

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy + Recall

6️⃣ Model Training

Trained with class weights to handle imbalance

Batch size: 32

Epochs: 20

Validation used to monitor overfitting

Visualization:

Training & validation accuracy and loss curves to monitor progress

7️⃣ Evaluation

Metrics calculated:

Accuracy

Recall

Precision

F1-Score

Confusion Matrix: Visualizes correct and incorrect predictions

Sample predictions: Displayed X-ray images with true label, predicted label, and confidence

Results:

Model performed well despite imbalanced dataset

Augmentation + class weighting improved minority class detection

8️⃣ Deployment

Model saved as chest_xray_cnn_model.h5

Can be used in Streamlit app for interactive predictions on new X-ray images

9️⃣ Key Takeaways

Handling imbalanced datasets is crucial in medical imaging.

Data augmentation improves generalization and prevents overfitting.

CNNs can achieve high accuracy in medical image classification.

Visualizing predictions and confidence scores makes the model interpretable and trustworthy.

🔗 Project Links

Kaggle Notebook: https://www.kaggle.com/code/rehamfaw/chest-x-ray-classification-using-cnn

