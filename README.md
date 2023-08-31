# BMI Classification Model Documentation

This documentation provides an overview of the code that demonstrates the creation and evaluation of a classification model for predicting BMI (Body Mass Index) categories using machine learning techniques. The code includes data preprocessing, model training, and evaluation using various classification algorithms. The dataset used in this project is stored in the "bmi.csv" file and contains features like height, weight, and gender.

## Table of Contents

1. **Import Libraries**: All the required libraries, including numpy, pandas, matplotlib, seaborn, Scipy, Sklearn, and imblearn, are imported for data manipulation, visualization, and machine learning.

2. **Data Exploration**: Initial exploration of the dataset, including creating a 'Class' column by mapping the 'Index' values to class names and displaying basic information about the dataset using functions like `info()`, `nunique()`, and `describe()`.

3. **Exploratory Data Analysis (EDA)**: Visual analysis of the dataset using various plots. This section includes pie charts for the distribution of the target variable, histograms, scatter plots, box plots, bar plots, and pair plots to understand the relationships between features.

4. **Data Preprocessing**: Preprocessing steps, such as converting the 'Gender' feature to numerical values (1 for 'Male' and 0 for 'Female'), splitting the dataset into training and testing sets, and performing feature scaling using StandardScaler.

5. **Model Training and Evaluation**:
   - **Logistic Regression (LR)**: Training a logistic regression model, performing hyperparameter tuning using RandomizedSearchCV, making predictions, and evaluating the model's performance using accuracy, precision, F1-score, confusion matrix, and classification report.
   - **Decision Tree (DT)**: Building a decision tree classifier, tuning hyperparameters with RandomizedSearchCV, making predictions, and evaluating the model's performance.
   - **Random Forest (RF)**: Creating a random forest classifier, hyperparameter tuning with RandomizedSearchCV, making predictions, and evaluating the model's performance.
   - **Support Vector Machine (SVM)**: Training a support vector machine classifier, hyperparameter tuning with RandomizedSearchCV, making predictions, and evaluating the model's performance.

6. **Comparison**: A comparison of the evaluation metrics (accuracy, precision, F1-score) for all the trained models is presented in a heatmap using seaborn.

## Conclusion

This code demonstrates the entire process of building and evaluating a BMI classification model using various machine learning algorithms. It covers data exploration, visualization, preprocessing, model training, hyperparameter tuning, prediction, and performance evaluation. By comparing the results of different models, you can determine which algorithm performs best for predicting BMI categories based on the given dataset. The documentation provides insights into the steps taken and the results obtained throughout the entire workflow.
