# Advertising Sales Prediction using Machine Learning

## Project Overview

This project is an end-to-end Machine Learning application that predicts product sales based on advertising expenditure.

The project uses Linear Regression to predict Sales using three advertising features:

- TV Advertising
- Radio Advertising
- Newspaper Advertising

A Streamlit web application is developed to allow users to enter advertising values and get predicted sales.

---

## Problem Statement

Advertising plays an important role in increasing product sales. Different advertising platforms such as TV, Radio, and Newspaper can affect sales.

The objective of this project is to build a Machine Learning model that predicts Sales based on the advertising budget spent on TV, Radio, and Newspaper.

---

## Objectives

- Collect and prepare advertising data.
- Perform data preprocessing.
- Select input features and target variable.
- Build a Linear Regression model.
- Evaluate the model using different evaluation metrics.
- Save the trained Machine Learning model.
- Develop a Streamlit user interface.
- Predict sales based on user input.
- Interpret the prediction results.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- VS Code

---

## Dataset Description

The dataset contains advertising expenditure and corresponding sales values.

### Features

| Feature | Description |
|---|---|
| TV | Advertising expenditure on TV |
| Radio | Advertising expenditure on Radio |
| Newspaper | Advertising expenditure on Newspaper |

### Target Variable

**Sales** - The output variable that represents product sales.

---

## Machine Learning Model

### Linear Regression

Linear Regression is used to predict the Sales value from the three input features:

- TV
- Radio
- Newspaper

The dataset is divided into training and testing data.

The model is trained using the training dataset and evaluated using the testing dataset.

---

## Model Evaluation

The trained model was evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| MAE | 1.734 |
| MSE | 3.473 |
| RMSE | 1.864 |
| R² Score | 0.886 |

### Interpretation

- **MAE** measures the average absolute difference between actual and predicted sales.
- **MSE** measures the average squared prediction error.
- **RMSE** represents the prediction error in the same general scale as the target.
- **R² Score** indicates how well the model explains variation in the target variable.

The obtained R² score is **0.886** on the test split used in this project.

---

## Model Saving

The trained model is saved using Joblib.

Model file:

`advertising_sales_model.pkl`

The saved model is loaded by the Streamlit application for making predictions.

---

## Streamlit Application

A user-friendly Streamlit interface is developed for prediction.

The user enters:

1. TV Advertising Budget
2. Radio Advertising Budget
3. Newspaper Advertising Budget

After clicking the **Predict Sales** button, the application displays the predicted sales value.

---

## Prediction and Interpretation

The application takes the user's advertising values as input and sends them to the trained Linear Regression model.

The model then generates the predicted Sales value.

A higher predicted Sales value indicates the model estimates higher sales for the given advertising inputs.

---

## Project Structure

```text
ML_Project/
│
├── train_model.py
├── app.py
├── advertising_dataset.csv
├── advertising_sales_model.pkl
└── README.md