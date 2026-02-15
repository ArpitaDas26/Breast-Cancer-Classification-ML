 Breast Cancer Classification using Machine Learning 

Project Overview:
This project aims to develop and compare multiple machine learning classification models to predict whether a breast tumor is malignant or benign using structured medical features. The models are evaluated using standard performance metrics to identify the most effective classifier.

The objective is to evaluate different classification algorithms and select the best-performing model based on performance metrics.

Dataset:
- Dataset: Breast Cancer Dataset (sklearn)
- Total Samples: 569
- Features: 30 numerical features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

The dataset was split into training and testing sets using an 80-20 split.


Preprocessing:
- Train-Test Split
- Feature Scaling using StandardScaler
- Feature scaling was performed using StandardScaler to ensure that all input features contribute equally to the model training process.


Models Implemented:

The following 6 Machine Learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost


Evaluation Metrics:

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1-Score
- MCC (Matthews Correlation Coefficient)
- Confusion Matrix


 Best Model:

Based on comparative evaluation across all performance metrics, Logistic Regression demonstrated the most balanced and consistent performance, achieving the highest Accuracy and AUC score among the tested models.

Test Accuracy: ~98%

Confusion Matrix:
[[41 1]
[ 1 71]]
The model correctly classified the majority of malignant and benign cases with very high precision and recall.


Project Structure:
Breast-Cancer-Classification-ML

model_training.ipynb # Model training and comparison
model_evaluation.py # Evaluation on test dataset
app.py # Streamlit web application
best_model.pkl # Saved trained model
test_data.csv # Test dataset
requirements.txt # Required Python libraries
README.md # Project documentation

Streamlit Application:

A simple Streamlit app was created to-
- Upload a test dataset (CSV)
- Generate predictions using the saved model
- Display classification results

To run the application locally:
1. Install dependencies:
   pip install -r requirements.txt
2. Run the Streamlit app:
   streamlit run app.py


Installation:
Install dependencies using-
pip install -r requirements.txt

Conclusion:
This study demonstrates the effectiveness of classical machine learning algorithms in medical classification tasks. The results indicate that Logistic Regression provides strong predictive performance for breast cancer diagnosis on this dataset. The deployment of the model using Streamlit further illustrates its practical usability in an interactive environment.
