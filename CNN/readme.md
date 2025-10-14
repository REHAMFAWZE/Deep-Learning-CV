🧠 CNN for MNIST Classification
📋 Project Overview

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.
The model is developed with TensorFlow/Keras and includes multiple configurations for experimentation with activation functions, optimizers, learning rates, dropout, and batch sizes.

🧩 Project Pipeline

The notebook includes a complete workflow for:

Loading and preprocessing the MNIST dataset

Building a customizable CNN architecture

Training with various hyperparameters

Evaluating model performance and visualizing results

🏗️ Model Architecture

The CNN model consists of:

Two convolutional layers (32 and 64 filters)

One max pooling layer

Optional dropout layers for regularization

Fully connected layer with 128 neurons

Output layer with 10 neurons (for digits 0–9)

⚙️ Key Features
🧹 Data Preprocessing

Normalization (scaling pixel values between 0–1)

Standardization (mean subtraction and standard deviation division)

One-hot encoding for labels

Train/validation split (90% / 10%)

🧠 Model Configuration

The custom Model() function allows flexibility in:

Activation function (default: ReLU)

Dropout rates at two layers

Filter sizes and counts

🔧 Training Configuration

The Compile_Train() function supports:

Multiple optimizers (SGD, Adam, RMSprop)

Adjustable learning rate, batch size, and epochs

Automatic plotting of training/validation accuracy

🧪 Experiments Conducted

Baseline Model:

No dropout

Optimizer: SGD

Activation: ReLU

Batch size: 64

Epochs: 5

Learning Rate Comparison:
Tested learning rates → [0.01, 0.001, 0.0001]

📊 Results
Metric	Value
Training Accuracy	~99.6%
Validation Accuracy	~99.0%
Test Accuracy	99.11%

✅ Excellent performance achieved with simple configurations.

💻 Usage
🔹 Basic Model Training
model = Model(activation_fn='relu', dropout_rate_1=0.0, dropout_rate_2=0.0)
acc = Compile_Train(model, optimizer_choice='sgd', learning_rate=0.01, 
                    momentum=0.9, batch_size=64, epochs=5)

🔹 Hyperparameter Testing
for lr in [0.01, 0.001, 0.0001]:
    model = Model(activation_fn='relu', dropout_rate_1=0.0, dropout_rate_2=0.0)
    Compile_Train(model, optimizer_choice='sgd', learning_rate=lr, 
                  momentum=0.9, batch_size=64, epochs=5)

📦 Requirements

TensorFlow

Scikit-learn

NumPy

Matplotlib

📁 File Structure
📦 CNN_MNIST_Classification
 ┣ 📜 CNN_MNIST.ipynb
 ┣ 📜 README.md
 ┗ 📂 outputs (optional for saving plots)


Data loading and preprocessing

Model definition (Model())

Training and evaluation (Compile_Train())

Experimental configurations

Visualization of training progress

⚡ Technical Notes

GPU acceleration used (NVIDIA T4 on Google Colab)

Includes comprehensive logging and progress tracking

Automatically generates accuracy/loss plots for each run

Supports reproducible experiments through consistent configurations

🧩 Summary

This implementation provides a flexible and modular framework for experimenting with CNN architectures and hyperparameters on the MNIST dataset, achieving high accuracy with minimal complexity.

👩‍💻 Author

Reham Fawzy Sayed
🎓 Computer Science & AI Student | Deep Learning Enthusiast
📧 [remonaaa734@gmail.com]
]
🌐 [https://rehamfawze.github.io/Portfolio/]
