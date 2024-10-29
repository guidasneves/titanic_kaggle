# Titanic Kaggle Competition
## About
The [Titanic challenge](https://www.kaggle.com/competitions/titanic/overview) on Kaggle is a competition in which the task is to predict the survival or the death of a given passenger based on a set of variables describing him such as his age, his sex, or his passenger class on the boat.

## Table of contents
* [Titanic Kaggle Competition](#titanic_kaggle_competition)
  * [About](#about)
  * [Table of contents](#table_of_contents)
  * [Data Preparation](#data_preparation)
  * [Model Evaluate](#model_evaluate)
    * [Logistic Regression](logistic_regression.ipynb)
    * [Neural Network](neural_network.ipynb)
    * [Random Forest](random_forest.ipynb)
    * [XGBoost](xgboost.ipynb)
  * [Error Analysis](#error_analysis)
  * [Final Model](#final_model)
  * [Entry](#entry)

## Data Preparation
```train['Sex_b'] = train['Sex'].map(lambda x: 1 if x == 'female' else 0)```
```test['Sex_b'] = test['Sex'].map(lambda x: 1 if x == 'female' else 0)```
