"""
imbalance_handler.py

This script demonstrates the engineering logic and mathematical workflow for handling 
extremely imbalanced datasets (e.g., 90:10 clinical anomalies) using Synthetic Minority 
Over-sampling Technique (SMOTE) and Scikit-Learn.

Designed for interviewing preparation, it highlights:
1. Synthetic generation of highly imbalanced data.
2. The critical principle of splitting data BEFORE applying oversampling (preventing data leakage).
3. The geometric and dimensional changes in data arrays at each step.
4. Model training and appropriate evaluation metrics for imbalanced datasets.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def run_pipeline():
    print("=" * 80)
    print("STEP 1: Generating Synthetic Imbalanced Dataset")
    print("=" * 80)
    
    # Generate a classification dataset:
    # - n_samples=10000: Total number of rows in the dataset
    # - n_features=20: Column dimensionality of the feature matrix
    # - weights=[0.9, 0.1]: Enforces a 90% majority class and 10% minority class distribution
    # - random_state=42: Seed to ensure deterministic results across runs
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # --- DIMENSIONAL ANNOTATION ---
    # X shape is (n_samples, n_features) -> (10000, 20)
    # y shape is (n_samples,) -> (10000,)
    print(f"Generated raw feature matrix X shape: {X.shape} (10,000 observations, 20 features)")
    print(f"Generated raw target vector y shape:  {y.shape} (10,000 labels)")
    
    # Show initial class distribution using Pandas Series value counts
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    class_pct = y_series.value_counts(normalize=True) * 100
    
    print("\nInitial Label Value Counts (90:10 ratio target):")
    for label, count in class_counts.items():
        print(f"  Class {label} (Clinical Anomalies if 1): {count:5d} samples ({class_pct[label]:.1f}%)")

    print("\n" + "=" * 80)
    print("STEP 2: Train-Test Split (Crucial Step: BEFORE Oversampling)")
    print("=" * 80)
    
    # --- INTERVIEW TALKING POINT: PREVENTING DATA LEAKAGE ---
    # NEVER apply SMOTE or other resampling methods before train/test splitting.
    # Reason: SMOTE synthesizes new data points by interpolating between neighboring minority class points.
    # If applied prior to the split, information from validation/test samples will leak into the
    # training set because SMOTE will use test/validation points as neighbors to construct new train points.
    # This leads to severe data leakage and artificially inflated performance evaluations.
    # The test set must represent the raw, unaltered real-world distribution (90:10).
    
    # We use 'stratify=y' to preserve the 90:10 class ratio in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- DIMENSIONAL ANNOTATION ---
    # X_train shape: (8000, 20) | y_train shape: (8000,) -> 80% of total samples
    # X_test shape:  (2000, 20) | y_test shape:  (2000,) -> 20% of total samples
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    
    train_counts = pd.Series(y_train).value_counts()
    train_pct = pd.Series(y_train).value_counts(normalize=True) * 100
    print("\nTraining Set Class Distribution (Pre-SMOTE):")
    for label, count in train_counts.items():
        print(f"  Class {label}: {count:5d} samples ({train_pct[label]:.1f}%)")

    print("\n" + "=" * 80)
    print("STEP 3: Applying SMOTE to Training Arrays")
    print("=" * 80)
    
    # SMOTE (Synthetic Minority Over-sampling Technique)
    # How it works:
    # 1. Selects a minority class sample x_i.
    # 2. Identifies its k-nearest neighbors in the minority class (usually k=5).
    # 3. Randomly chooses one neighbor x_zi.
    # 4. Computes the difference: diff = x_zi - x_i.
    # 5. Multiplies this difference by a random number lambda in range [0, 1] and adds it to x_i:
    #    x_new = x_i + lambda * (x_zi - x_i)
    # This creates a new synthetic minority sample along the line segment joining x_i and x_zi.
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # --- DIMENSIONAL ANNOTATION ---
    # Majority class (Class 0) in training set: 7200 samples
    # Minority class (Class 1) in training set: 800 samples
    # SMOTE synthesizes: 7200 - 800 = 6400 new Class 1 samples
    # Post-SMOTE dimensions:
    # X_train_res shape becomes (14400, 20) (i.e. 7200 majority + 7200 minority)
    # y_train_res shape becomes (14400,)
    
    print(f"Resampled X_train_res shape: {X_train_res.shape} (Expanded from {X_train.shape})")
    print(f"Resampled y_train_res shape: {y_train_res.shape} (Expanded from {y_train.shape})")
    
    resampled_counts = pd.Series(y_train_res).value_counts()
    resampled_pct = pd.Series(y_train_res).value_counts(normalize=True) * 100
    print("\nTraining Set Class Distribution (Post-SMOTE):")
    for label, count in resampled_counts.items():
        print(f"  Class {label}: {count:5d} samples ({resampled_pct[label]:.1f}%)")
        
    print("\n" + "=" * 80)
    print("STEP 4: Training Baseline Logistic Regression Model")
    print("=" * 80)
    
    # Train Logistic Regression on the now-balanced training splits
    # We increase max_iter to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_res, y_train_res)
    print("Logistic Regression model trained successfully on balanced (50:50) data.")
    
    print("\n" + "=" * 80)
    print("STEP 5: Model Evaluation on Unseen Test Data")
    print("=" * 80)
    
    # Predict labels on the unseen test set (2,000 samples with original 90:10 ratio)
    # This evaluates how the model generalizes to the true clinical anomaly distribution.
    y_pred = model.predict(X_test)
    
    # --- DIMENSIONAL ANNOTATION ---
    # y_test shape: (2000,)
    # y_pred shape: (2000,)
    print(f"Test targets shape: {y_test.shape} | Predictions shape: {y_pred.shape}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (Evaluated on Unseen Imbalanced Test Set):")
    print(classification_report(y_test, y_pred))
    
    # --- INTERVIEW TALKING POINT: METRIC INTERPRETATION ---
    # 1. Accuracy: A naive model that classifies everything as Class 0 would get 90% accuracy.
    #    Therefore, accuracy is not a reliable metric.
    # 2. Recall (Sensitivity) for Class 1: Out of all actual clinical anomalies, what percentage did we catch?
    #    Recall = TP / (TP + FN). In a medical context, maximizing recall is often primary.
    # 3. Precision for Class 1: Out of all predicted clinical anomalies, what percentage were correct?
    #    Precision = TP / (TP + FP). A low precision means many false alarms.
    # 4. F1-Score: The harmonic mean of precision and recall. It gives a balanced metric of overall class utility.
    
if __name__ == "__main__":
    run_pipeline()
