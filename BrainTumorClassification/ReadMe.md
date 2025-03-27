🧠 Brain Tumor Classification using Ensemble Model

This project focuses on brain tumor classification using an ensemble of deep learning models to improve accuracy and robustness. The ensemble combines InceptionV3, DenseNet, and VGG16, leveraging their strengths to classify MRI scans as tumor or non-tumor. Additionally, Explainable AI (XAI) techniques, such as SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-Agnostic Explanations), are integrated to enhance model interpretability.

✨ Features
     1.	Data Preprocessing: MRI images undergo skull stripping, cropping, resizing, CLAHE (Contrast 	Limited Adaptive Histogram Equalization), and normalization for enhanced feature extraction.
     2.	Deep Learning Models: Implements InceptionV3, DenseNet, and VGG16 for feature extraction and 	classification. ResNet50 was tested but excluded due to poor performance on the dataset.
     3.	Ensemble Learning: Combines multiple models using weighted averaging to improve classification 	performance.
     4.	Model Evaluation: Assesses performance using accuracy, F1-score, and confusion matrices.

🔍 Workflow
     1.	Data Preprocessing: MRI scans undergo skull stripping, enhancement, and normalization.
     2.	Model Training: Individual models (InceptionV3, DenseNet, and VGG16) are trained separately.
     3.	Ensemble Prediction: A weighted averaging approach is used to combine model predictions.
     4.	Evaluation: The ensemble’s performance is analyzed using classification metrics.

🏗️ Code Structure
     1.	Dataset Handling: Loads and preprocesses MRI images.
     2.	Preprocessing Pipeline: Applies skull stripping, cropping, CLAHE, and normalization.
     3.	Model Training: Defines and trains InceptionV3, DenseNet, and VGG16.
     4.	Ensemble Model: Aggregates predictions from multiple models.
     5.	Model Evaluation: Computes classification metrics and confusion matrices.
