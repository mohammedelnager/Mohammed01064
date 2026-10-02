# 🏠 House Price Prediction

A simple Machine Learning project that predicts house prices using **Random Forest Regression**.

## 📌 Project Overview

This project uses a small dataset containing information about houses and their prices.

The model learns the relationship between house features and the house price, then uses this knowledge to predict prices for new data.

## 📊 Features

The model uses three features:

* **Area** → House area
* **Rooms** → Number of rooms
* **Distance** → Distance from a specific location

The target variable is:

* **Price** → House price

## 🤖 Machine Learning Model

The project uses:

**Random Forest Regressor**

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The dataset is divided into:

* **Training data** → Used to train the model
* **Testing data** → Used to evaluate the model

The split is performed using:

```python
train_test_split()
```

with a test size of **20%**.

## 📈 Model Evaluation

The model is evaluated using three metrics:

### MSE — Mean Squared Error

Measures the average squared difference between the actual and predicted prices.

### RMSE — Root Mean Squared Error

The square root of MSE, giving the error in the same unit as the target price.

### R² Score

Measures how well the model explains the variation in house prices.

## ⭐ Feature Importance

Random Forest provides **Feature Importance**, which helps identify how much each feature contributes to the model's predictions.

```python
model.feature_importances_
```

## 📊 Visualizations

The project contains two visualizations:

### 1. Actual vs Predicted

A scatter plot comparing the actual house prices with the prices predicted by the model.

### 2. 3D Visualization

A 3D scatter plot showing the relationship between:

* Area
* Number of Rooms
* Price

## 🛠️ Technologies Used

* Python
* NumPy
* Scikit-learn
* Matplotlib
* Random Forest

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── index.html
├── house_price.py
├── README.md
└── requirements.txt
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the Python program:

```bash
python house_price.py
```

## 🎯 Learning Objectives

This project was created to practice the basic Machine Learning workflow:

**Data → Split → Model → Training → Prediction → Evaluation → Visualization**

It also provides practical experience with:

* Regression
* Random Forest
* Train/Test Split
* Model Evaluation
* Feature Importance
* Data Visualization

## 👨‍💻 Author

**Mohamed Sobhy**

Computer & Systems Engineering Student
Interested in Artificial Intelligence, Machine Learning and Deep Learning.
