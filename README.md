# Heart Disease Prediction Model

This repository contains a comprehensive machine learning project for predicting heart disease using clinical and lifestyle data. The project includes extensive exploratory data analysis (EDA), feature engineering, and multiple machine learning model implementations.

## 📊 Dataset

The project uses the **Heart Failure Prediction** dataset from Kaggle, which contains patient-level data with various clinical and lifestyle attributes. The dataset includes:

- **918 patients** with **12 features**
- **Target variable**: HeartDisease (binary: 0 = No, 1 = Yes)
- **Features**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

## 🎯 Project Overview

This project demonstrates a complete machine learning workflow:

1. **Data Loading and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Model Training and Evaluation**
5. **Model Persistence**

## 📈 Key Findings

### Data Quality
- **Dataset Size**: 918 patients, 12 features
- **Missing Values**: None detected
- **Class Distribution**: Relatively balanced (45.5% No Heart Disease, 54.5% Heart Disease)
- **Data Issues**: Zero values in RestingBP (1 case) and Cholesterol (172 cases) - imputed with medians

### Statistical Significance
All features show statistically significant relationships with heart disease (p < 0.05):
- **Numerical Features**: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- **Categorical Features**: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

### Key Insights
- **Age**: Moderate positive correlation with heart disease
- **MaxHR**: Moderate negative correlation with heart disease
- **ChestPainType**: 'ASY' (asymptomatic) most frequent in heart disease cases
- **ExerciseAngina**: Strong association with heart disease
- **ST_Slope**: 'Flat' and 'Down' slopes more prevalent in heart disease group

## 🛠️ Technical Implementation

### Data Preprocessing
- **Zero Value Handling**: Imputed 0 values in RestingBP and Cholesterol with median values
- **Outlier Analysis**: Identified outliers using IQR method
- **Feature Engineering**: 
  - Created AgeGroup feature by binning Age
  - One-hot encoded categorical variables
  - Generated synthetic data for augmentation

### Models Trained
1. **Logistic Regression** - Accuracy: 89.13%
2. **Random Forest** - Accuracy: 86.41%
3. **Gradient Boosting** - Accuracy: 86.96%
4. **XGBoost** - Accuracy: 85.33%

### Model Performance
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8913 | 0.9020 | 0.9020 | 0.9020 |
| Random Forest | 0.8641 | 0.8738 | 0.8824 | 0.8780 |
| Gradient Boosting | 0.8696 | 0.8750 | 0.8922 | 0.8835 |
| XGBoost | 0.8533 | 0.8788 | 0.8529 | 0.8657 |

## 📁 Project Structure

```
datacoach_ai_demo_2508/
├── heart_disease_colab.ipynb    # Main Jupyter notebook
├── README.md                    # This file
├── Untitled.ipynb              # Additional notebook
└── Model Files (generated):
    ├── log_reg_model.pkl       # Logistic Regression model
    ├── rf_model.pkl            # Random Forest model
    ├── gb_model.pkl            # Gradient Boosting model
    ├── xgb_model.pkl           # XGBoost model
    ├── *_predictions.txt       # Model predictions
    └── *_metrics.txt           # Model evaluation metrics
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

### Installation
1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```
3. Download the heart.csv dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
4. Place the dataset in the project directory
5. Open `heart_disease_colab.ipynb` in Jupyter Notebook

### Running the Analysis
1. Execute all cells in the notebook sequentially
2. The notebook will automatically:
   - Load and preprocess the data
   - Perform comprehensive EDA
   - Train multiple models
   - Save models and results

## 📊 EDA Components

The project includes a comprehensive 11-phase EDA process:

1. **Dataset Structure Analysis**
2. **Missing Values Analysis**
3. **Basic Statistical Summary**
4. **Univariate Analysis**
5. **Outlier Detection**
6. **Correlation Analysis**
7. **Bivariate Analysis**
8. **Multivariate Analysis**
9. **Feature Engineering Insights**
10. **Class Imbalance Check**
11. **Advanced EDA Enhancements**

## 🔧 Feature Engineering

### Implemented Features
- **AgeGroup**: Binned age categories for better pattern recognition
- **One-Hot Encoding**: Applied to all categorical variables
- **Synthetic Data Generation**: Created additional samples for model robustness

### Potential Enhancements
- Interaction terms between Oldpeak and ST_Slope
- Combined features for ChestPainType and ExerciseAngina
- Numerical transformations for skewed features

## 📈 Model Evaluation

Models are evaluated using multiple metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## 🎯 Results and Insights

### Best Performing Model
**Logistic Regression** achieved the highest accuracy (89.13%) with balanced precision and recall scores, making it the most reliable model for this dataset.

### Key Risk Factors Identified
1. **Age**: Higher age associated with increased heart disease risk
2. **Exercise Angina**: Strong predictor of heart disease
3. **ST Slope**: Abnormal slopes indicate higher risk
4. **Chest Pain Type**: Asymptomatic cases require special attention
5. **Max Heart Rate**: Lower rates associated with heart disease

## 🔮 Future Enhancements

1. **Hyperparameter Tuning**: Implement grid search or Bayesian optimization
2. **Ensemble Methods**: Combine multiple models for improved performance
3. **Deep Learning**: Explore neural network architectures
4. **Feature Selection**: Implement automated feature selection techniques
5. **Cross-Validation**: Add k-fold cross-validation for more robust evaluation
6. **Model Interpretability**: Add SHAP or LIME for model explanation

## 📝 License

This project is for educational and research purposes. The dataset is sourced from Kaggle and should be used in accordance with their terms of service.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the project.

## 📞 Contact

For questions or suggestions about this project, please open an issue in the repository.

---

**Note**: This project is designed for educational purposes and should not be used for actual medical diagnosis without proper validation and clinical expertise. 