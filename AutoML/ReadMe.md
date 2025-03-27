🤖 AutoML System for Supervised Learning

This AutoML (Automated Machine Learning) system is designed for Supervised Learning tasks, with a focus on transparency and interpretability through Explainable AI (XAI) techniques. The system automates various stages of the machine learning pipeline, including data preprocessing, feature selection, model selection, hyperparameter optimization, and XAI integration using SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-Agnostic Explanations).

✨ Features
     1.	Data Preprocessing: Handles irrelevant columns, duplicate values, constant values, missing 	values, and categorical encoding for structured datasets.
     2.	Model Selection: Supports Decision Trees, Random Forest, and Gradient Boosting for both 	classification and regression problems.
     3.	Hyperparameter Optimization: Implements Bayesian Optimization to fine-tune model parameters 	for improved performance.
     4.	Explainable AI (XAI): Integrates SHAP and LIME to provide interpretability, helping understand 	model predictions.

🔍 Workflow
     1.	Data Preprocessing: Cleans and prepares the dataset by handling missing values and categorical 	variables.
     2.	Model Selection: Chooses the best-performing model based on the problem type (classification 	or regression).
     3.	Hyperparameter Optimization: Fine-tunes model parameters using Bayesian Optimization.
     4.	Model Evaluation: Assesses performance using metrics like accuracy, F1-score, precision, 	recall (for classification) and R-squared, MAE, MSE, RMSE (for regression).
     5.	Explainability: Uses SHAP and LIME to interpret predictions and visualize feature importance.

🏗️ Code Structure
     1.	Data Preprocessing: Functions to clean, handle missing values, and encode categorical data.
     2.	Model Selection: Supports multiple algorithms for both classification and regression.
     3.	Hyperparameter Tuning: Uses Bayesian Optimization for performance enhancement.
     4.	Model Training & Evaluation: Trains models and evaluates them using relevant metrics.
     5.	XAI Integration: Uses SHAP and LIME for model interpretability and insights.
