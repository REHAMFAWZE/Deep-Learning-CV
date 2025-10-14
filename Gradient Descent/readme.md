🧠 Neural Network Gradient Descent Implementation with TensorBoard

This project demonstrates the implementation of gradient descent for a simple neural network (logistic regression) using TensorFlow/Keras, with full TensorBoard integration for visualization and analysis.

📘 Project Overview

The notebook implements a binary classification model to predict insurance purchase decisions based on age and affordability factors.
It covers the complete workflow:

✅ Data generation and preprocessing

✅ Neural network model creation

✅ Gradient descent optimization

✅ TensorBoard integration for training visualization

📊 Dataset

A synthetic dataset with 200 samples is generated containing the following features:

Feature	Description
age	Integer values between 20–65
affordability	Binary indicator (0 or 1) for financial capability
bought_insurance	Target variable (0 or 1) indicating purchase

Data characteristics:

Realistic probabilistic patterns (older individuals and those with higher affordability are more likely to buy insurance)

Semi-balanced classes: 106 (no) vs. 94 (yes)

🧩 Model Architecture

A simple single-layer neural network is used:

Sequential([
    Dense(1, activation="sigmoid", input_shape=(2,))
])


Configuration:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Epochs: 3000

Train-Test Split: 80–20

📈 TensorBoard Integration

TensorBoard is integrated to monitor training performance and visualize the learning process.

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch",
    profile_batch=0
)


Tracked Metrics:

Training loss and accuracy per epoch

Model graph

Weight histograms

Real-time training updates

⚙️ Key Features

🧩 Data Generation: Creates realistic, probabilistic data

🧠 Model Training: Demonstrates gradient descent with extensive logging

📊 Visualization: TensorBoard integration for performance tracking

🔍 Evaluation: Training progress over 3000 epochs

🚀 Usage

Run the notebook cells sequentially.

Launch TensorBoard to monitor progress:

tensorboard --logdir logs/fit


View real-time metrics, graphs, and loss/accuracy trends in TensorBoard.

🧰 Requirements

TensorFlow / Keras

pandas

numpy

scikit-learn

TensorBoard

Install dependencies:

pip install tensorflow pandas numpy scikit-learn tensorboard

💡 Key Insights

The neural network effectively learns the relationship between age, affordability, and insurance purchase.

TensorBoard offers valuable insights into model convergence and training dynamics.

The project serves as a clear, educational example of implementing gradient descent optimization in neural networks.

📚 Purpose

This implementation is designed for educational purposes — helping learners understand how simple neural networks can be trained and visualized with TensorBoard before moving to more complex architectures.
