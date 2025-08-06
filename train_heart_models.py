#!/usr/bin/env python3
"""
Heart Disease Prediction Model Training Script

This script trains machine learning models to predict heart disease based on clinical data.
It supports multiple model types and allows selection via command-line arguments.

Usage:
    python train_heart_models.py -m logreg -m xgboost -m svm
    python train_heart_models.py --models logreg rf gb xgb svm
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


def load_and_preprocess_data(data_path='heart.csv'):
    """Load and preprocess the heart disease dataset."""
    print(f"Loading data from {data_path}...")
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found.")
        sys.exit(1)
    
    # Handle missing values (0 values in RestingBP and Cholesterol)
    print("Preprocessing data...")
    
    # Replace 0 values with median for RestingBP and Cholesterol
    if (df['RestingBP'] == 0).any():
        median_bp = df[df['RestingBP'] > 0]['RestingBP'].median()
        df.loc[df['RestingBP'] == 0, 'RestingBP'] = median_bp
        print(f"Replaced {(df['RestingBP'] == 0).sum()} zero values in RestingBP with median: {median_bp}")
    
    if (df['Cholesterol'] == 0).any():
        median_chol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_chol
        print(f"Replaced {(df['Cholesterol'] == 0).sum()} zero values in Cholesterol with median: {median_chol}")
    
    # Encode categorical variables
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    print(f"Encoded categorical variables: {categorical_columns}")
    
    # Separate features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target distribution - No disease: {(y == 0).sum()}, Heart disease: {(y == 1).sum()}")
    
    return X, y


def get_model(model_name):
    """Get a model instance by name."""
    models = {
        'logreg': LogisticRegression(random_state=42, max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'xgb': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'svm': SVC(random_state=42, probability=True)
    }
    
    if model_name not in models:
        available_models = ', '.join(models.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")
    
    return models[model_name]


def train_model(model, X_train, y_train, model_name):
    """Train a model and return it."""
    print(f"\nTraining {model_name.upper()} model...")
    model.fit(X_train, y_train)
    print(f"{model_name.upper()} model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_pred


def print_metrics(model_name, metrics):
    """Print evaluation metrics for a model."""
    accuracy, precision, recall, f1 = metrics[:4]
    print(f"\n{model_name.upper()} Model Evaluation:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")


def save_model_results(model, model_name, metrics, y_test, y_pred):
    """Save model, predictions, and metrics to files."""
    # Save model
    model_filename = f"{model_name}_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {model_filename}")
    
    # Save predictions
    pred_filename = f"{model_name}_predictions.txt"
    with open(pred_filename, "w") as f:
        for i in range(len(y_pred)):
            f.write(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}\n")
    print(f"  Predictions saved to: {pred_filename}")
    
    # Save metrics
    metrics_filename = f"{model_name}_metrics.txt"
    accuracy, precision, recall, f1 = metrics[:4]
    with open(metrics_filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    print(f"  Metrics saved to: {metrics_filename}")


def main():
    """Main function to parse arguments and train selected models."""
    parser = argparse.ArgumentParser(
        description="Train heart disease prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_heart_models.py -m logreg -m xgboost
  python train_heart_models.py --models logreg rf gb xgb svm
  python train_heart_models.py -m svm

Available models:
  logreg  - Logistic Regression
  rf      - Random Forest
  gb      - Gradient Boosting
  xgb     - XGBoost
  svm     - Support Vector Machine
        """
    )
    
    parser.add_argument(
        '-m', '--models',
        action='append',
        choices=['logreg', 'rf', 'gb', 'xgb', 'svm'],
        help='Model(s) to train. Can be used multiple times to train multiple models.',
        required=True
    )
    
    parser.add_argument(
        '--data',
        default='heart.csv',
        help='Path to the heart disease dataset (default: heart.csv)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size as a fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Remove duplicates while preserving order
    selected_models = list(dict.fromkeys(args.models))
    
    print("Heart Disease Prediction Model Training")
    print("=" * 50)
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Data file: {args.data}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(args.data)
    
    # Split data
    print(f"\nSplitting data (train: {1-args.test_size:.1%}, test: {args.test_size:.1%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate selected models
    results = {}
    
    for model_name in selected_models:
        try:
            # Get model instance
            model = get_model(model_name)
            
            # Train model
            trained_model = train_model(model, X_train, y_train, model_name)
            
            # Evaluate model
            metrics = evaluate_model(trained_model, X_test, y_test)
            accuracy, precision, recall, f1, y_pred = metrics
            
            # Print results
            print_metrics(model_name, metrics)
            
            # Save results
            print(f"  Saving {model_name.upper()} results...")
            save_model_results(trained_model, model_name, metrics, y_test, y_pred)
            
            # Store results for summary
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            print(f"\nError training {model_name.upper()} model: {e}")
            continue
    
    # Print summary
    if results:
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1 Score':<8}")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            print(f"{model_name.upper():<15} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<11.4f} {metrics['recall']:<8.4f} "
                  f"{metrics['f1']:<8.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing model: {best_model[0].upper()} "
              f"(Accuracy: {best_model[1]['accuracy']:.4f})")
        
        print(f"\nAll models saved with predictions and metrics.")
        print("Training completed successfully!")
    else:
        print("\nNo models were successfully trained.")
        sys.exit(1)


if __name__ == "__main__":
    main()