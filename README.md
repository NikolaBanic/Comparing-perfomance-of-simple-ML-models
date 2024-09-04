# Comparing Machine Learning Models for Satellite Image Classification

## Project Overview

- **Data Preparation:** The dataset is loaded from a CSV file, and features are normalized using `StandardScaler`. The data is split into training, validation, and test sets, with a 70-15-15% distribution.
  
- **Models Evaluated:**
  - Logistic Regression
  - Random Forest (with and without hyperparameter tuning)
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - AdaBoost

- **Hyperparameter Tuning:** Random Forest parameters are optimized using `RandomizedSearchCV` to find the best performing model.

- **Performance Metrics:**
  - Accuracy on validation and test sets
  - Precision, Recall, F1 Score
  - Confusion matrices for visualizing classification performance
  - Number of misclassifications per model and class

## Visualizations

- **Data Distribution:** Visualized using KDE plots to understand the distribution of feature values.
- **Confusion Matrix:** Heatmaps are generated to display the confusion matrices of each model's predictions on the test set.
- **Misclassifications:** Bar plots illustrate the number of misclassifications for each model across different classes.

## Project Structure

- `data/`: Directory for storing datasets.
- `models/`: Directory for saving trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `results/`: Directory for storing output files like confusion matrices, misclassification plots, and the model performance comparison CSV.
- `scripts/`: Python scripts for data processing, model training, and evaluation.
