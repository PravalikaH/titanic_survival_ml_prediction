# Titanic Survival Prediction
# Titanic Survival Prediction — ML Classification

This repository contains my submission for **Project 02** in the AI & ML Internship Program at Pluto Academy.

---

## 📌 Project Overview
The objective is to analyze historical passenger records from the Titanic disaster and build a machine learning model that accurately predicts survival.

- **Dataset Source:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Problem Type:** Binary Classification

---

## 🛠️ Workflow & Methodology

1. **Data Preprocessing:**
   - Imputed missing `Age` values using the median to avoid outlier skewness.
   - Imputed missing `Embarked` entries using the mode (`'S'`).
   - Dropped the `Cabin` column (>77% missing data).
   - Encoded `Sex` (0 for male, 1 for female) and one-hot encoded `Embarked`.

2. **Feature Engineering:**
   - Created `FamilySize` (`SibSp + Parch + 1`) and `IsAlone` binary indicators.
   - Dropped identifier columns (`PassengerId`, `Name`, `Ticket`).
   - Conducted an 80/20 stratified train-test split and normalized features with `StandardScaler`.

3. **Model Training & Comparison:**
   - Trained three distinct classification models: Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).

---

## 📊 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7989** | **0.8367** | 0.5942 | 0.6949 |
| K-Nearest Neighbors | **0.7989** | 0.7705 | **0.6812** | **0.7231** |
| Logistic Regression | 0.7933 | 0.7500 | **0.6957** | 0.7218 |

---

## 🏆 Best Model Analysis & Conclusion

1. Random Forest and K-Nearest Neighbors tied for the highest overall accuracy at **79.89%**, outperforming Logistic Regression (**79.33%**).
2. Random Forest achieved the highest precision score of **83.67%**, producing the fewest false-positive survival predictions on the test set.
3. While Random Forest had lower recall (**59.42%**) compared to KNN (**68.12%**) and Logistic Regression (**69.57%**), its ensemble tree structure captured non-linear interactions across features effectively.
4. Setting a tree depth constraint helped regulate variance and prevented the ensemble from overfitting on noisy passenger-level data.
5. Random Forest is selected as the preferred model due to its high accuracy, superior precision, and robust generalization on tabular data.

---

## 🔗 Project Links
- **Google Colab Notebook:** [View Notebook](https://colab.research.google.com/drive/1vm7ycBExQvMeszwbAkjSlgJXeYHEBbl_?usp=sharing)
- **GitHub Repository:** [https://github.com/PravalikaH/titanic_survival_ml_prediction](https://github.com/PravalikaH/titanic_survival_ml_prediction)
