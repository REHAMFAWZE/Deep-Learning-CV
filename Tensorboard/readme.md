🧠 Neural Network Implementation with TensorBoard
📋 Project Overview

This project demonstrates the implementation of a Neural Network for binary classification using TensorFlow/Keras, integrated with TensorBoard for real-time training visualization and monitoring.
The model predicts insurance purchasing behavior based on age and affordability, showing how simple neural architectures can be effectively analyzed using TensorBoard tools.

⚙️ Key Features

🧮 Synthetic Data Generation — creates realistic patterns between input features and target.

🧠 Neural Network Architecture — single-layer perceptron using sigmoid activation for binary output.

📊 TensorBoard Integration — visualizes training metrics, histograms, and computation graph.

🔁 Training Process — runs for 3000 epochs with gradient descent optimization.

✅ Binary Classification Task — predicts whether a customer buys insurance.

🧾 Dataset

The synthetic dataset consists of 200 samples with 3 variables:

Feature	Type	Description
age	Integer	Values between 20–65
affordability	Binary	0 (low affordability) or 1 (high affordability)
bought_insurance	Target	0 (no) or 1 (yes)
📈 Data Characteristics

Older individuals are more likely to purchase insurance.

Higher affordability increases purchase probability.

The target variable is semi-balanced (106:94 ratio).

🧩 Model Architecture
Sequential([
    Dense(1, activation="sigmoid", input_shape=(2,))
])

Configuration:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

🎛️ TensorBoard Integration

Comprehensive TensorBoard logging is implemented using:

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch",
    profile_batch=0
)

Tracked Metrics:

Training loss and accuracy

Weight histograms

Computation graph

Epoch-wise performance curves

🚀 Model Training

Epochs: 3000

Performance: Accuracy improves steadily, reaching 75–80%

Behavior: Smooth convergence with stable loss reduction

💻 Usage

Run the notebook sequentially from top to bottom.

Start TensorBoard to monitor training:

tensorboard --logdir logs/fit


Open the TensorBoard link (typically http://localhost:6006/
)

View real-time metrics, loss curves, and computation graph.

📦 Requirements

TensorFlow 2.x

NumPy

Pandas

Scikit-learn

TensorBoard

Install dependencies:

pip install tensorflow numpy pandas scikit-learn tensorboard

🧠 Key Insights

The model successfully learns meaningful relationships between age, affordability, and insurance purchase.

TensorBoard provides deep insights into the training dynamics and model behavior.

Demonstrates the power of visual analytics in neural network development.

Serves as a strong educational example and foundation for more advanced architectures.

