# Heart Disease Model Training Script

This repository contains a command-line Python script `train_heart_models.py` that implements a streamlined version of the machine learning pipeline from `heart_disease_colab.ipynb`. The script allows you to train and evaluate multiple heart disease prediction models by specifying which models to use via command-line flags.

## Features

- **Multiple Model Support**: Train any combination of 5 different machine learning models
- **Command-Line Interface**: Easy-to-use CLI with flexible model selection
- **Comprehensive Output**: Generates model files, predictions, and evaluation metrics
- **Data Preprocessing**: Automatic handling of missing values and categorical encoding
- **Performance Summary**: Comparison table of all trained models

## Supported Models

- `logreg` - Logistic Regression
- `rf` - Random Forest
- `gb` - Gradient Boosting
- `xgb` - XGBoost
- `svm` - Support Vector Machine

## Installation

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

### Basic Usage

Train a single model:
```bash
python train_heart_models.py -m logreg
```

Train multiple models (as specified in the issue):
```bash
python train_heart_models.py -m logreg -m xgb -m svm
```

Train all available models:
```bash
python train_heart_models.py -m logreg -m rf -m gb -m xgb -m svm
```

### Advanced Options

```bash
python train_heart_models.py -m logreg -m xgb \
    --data custom_data.csv \
    --test-size 0.3 \
    --random-state 123
```

### Command-Line Options

- `-m, --models`: Specify which models to train (can be used multiple times)
- `--data`: Path to the dataset file (default: heart.csv)
- `--test-size`: Test set size as a fraction (default: 0.2)
- `--random-state`: Random state for reproducibility (default: 42)
- `-h, --help`: Show help message

## Output Files

For each trained model, the script generates:

1. **Model file**: `{model_name}_model.pkl` - Serialized trained model
2. **Predictions file**: `{model_name}_predictions.txt` - Test set predictions vs actual values
3. **Metrics file**: `{model_name}_metrics.txt` - Evaluation metrics (accuracy, precision, recall, F1)

## Example Output

```
Heart Disease Prediction Model Training
==================================================
Selected models: logreg, xgb, svm
Data file: heart.csv
Test size: 0.2
Random state: 42

Loading data from heart.csv...
Dataset loaded successfully: 918 rows, 12 columns
Preprocessing data...
Features shape: (918, 11), Target shape: (918,)

Training LOGREG model...
LOGREG Model Evaluation:
  Accuracy:  0.8424
  Precision: 0.8476
  Recall:    0.8725
  F1 Score:  0.8599

[... training other models ...]

==================================================
TRAINING SUMMARY
==================================================
Model           Accuracy   Precision   Recall   F1 Score
------------------------------------------------------------
LOGREG          0.8424     0.8476      0.8725   0.8599  
XGB             0.8424     0.8763      0.8333   0.8543  
SVM             0.7065     0.7264      0.7549   0.7404  

Best performing model: LOGREG (Accuracy: 0.8424)
```

## Data Requirements

The script expects a CSV file with the following columns:
- Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS
- RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

The script automatically handles:
- Missing values (zero values in RestingBP and Cholesterol)
- Categorical variable encoding
- Train/test split with stratification

## Testing

Run the test suite to verify functionality:
```bash
python test_train_script.py
```

## Original Notebook

The original implementation and detailed analysis can be found in `heart_disease_colab.ipynb`.