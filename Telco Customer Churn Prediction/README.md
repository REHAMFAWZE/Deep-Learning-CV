📊 Telco Customer Churn Prediction - Deep Learning
🧠 Overview

This project implements a Deep Learning solution to predict customer churn using the Telco Customer Churn Dataset.
It focuses on analyzing customer behavior and identifying the key factors contributing to customer attrition in the telecommunications industry.

📂 Dataset

The dataset used is the Telco Customer Churn Dataset from Kaggle
, containing 7,043 customer records with 21 features, including:

Demographic Information: gender, senior citizen status, partner, dependents

Service Information: phone service, multiple lines, internet service type

Service Features: online security, online backup, device protection, tech support

Account Information: contract type, paperless billing, payment method

Charges: monthly charges, total charges

Target Variable: churn (Yes/No)

🧩 Project Structure
1️⃣ Data Loading and Initial Exploration

Mount Google Drive for dataset access

Load and inspect data structure

Perform basic exploratory data analysis (EDA)

2️⃣ Data Preprocessing

Handle missing values in the TotalCharges column

Convert data types appropriately

Remove customerID (non-informative identifier)

Detect and remove duplicates

Verify overall data quality

3️⃣ Exploratory Data Analysis (EDA)

Generate statistical summaries for numerical features

Analyze categorical distributions

Visualize data using pie charts and histograms

Identify correlations and feature relationships

4️⃣ Feature Analysis
🔹 Categorical Features (16):

Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, Churn

🔹 Numerical Features (4):

SeniorCitizen, tenure, MonthlyCharges, TotalCharges

📈 Key Findings from EDA
🧹 Data Quality Issues Addressed

TotalCharges column converted from object → numeric

11 missing values handled using linear interpolation

22 duplicate records identified and removed

📊 Statistical Insights

SeniorCitizen: binary (0/1), slightly imbalanced distribution

tenure: continuous, semi-normal distribution

MonthlyCharges: continuous, near-normal distribution

TotalCharges: continuous, right-skewed distribution

⚙️ Technical Requirements
🧾 Libraries Used

NumPy, Pandas → data manipulation

Matplotlib, Seaborn → data visualization

Scikit-learn → preprocessing & evaluation

Warnings → suppressing unnecessary warnings

🧰 Environment

Python: 3.11.13

Platform: Google Colab (GPU enabled - NVIDIA Tesla T4)

Notebook Format: Jupyter Notebook (.ipynb)

🚀 Next Steps

This notebook establishes a solid foundation for:

Feature engineering & encoding categorical variables

Data normalization and scaling

Building and training Deep Learning models for churn prediction

Model evaluation, tuning, and optimization

💼 Business Value

This project enables telecom companies to:

Identify customers at high risk of churn

Understand key drivers of customer attrition

Design targeted retention campaigns

Improve customer lifetime value (CLV)

🧠 Usage

Clone the repository:

git clone https://github.com/yourusername/Telco-Customer-Churn-DeepLearning.git


Open the notebook in Google Colab or Jupyter.

Update the dataset path in the notebook to your Google Drive path.

Run all cells sequentially from top to bottom.

📝 Note

This notebook currently covers:

Data loading

Data preprocessing

Exploratory data analysis

The deep learning modeling phase will follow in the next version, including model design, training, and performance evaluation.

👩‍💻 Author

Reham Fawzy Sayed
📍 AI & Computer Science Student | Deep Learning & Computer Vision Enthusiast
📧 [remonaaa734@gmail.com
]
🌐 [https://rehamfawze.github.io/Portfolio/]
