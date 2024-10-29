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
Transforming the categorical sex feature into a binary feature.
```Python
train['Sex_b'] = train['Sex'].map(lambda x: 1 if x == 'female' else 0)
test['Sex_b'] = test['Sex'].map(lambda x: 1 if x == 'female' else 0)
```

Transforming the rest of the features from the training and test datasets, so that both datasets have the same transformation.
```Python
train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)

train['Cabin_null'] = train['Cabin'].isnull().astype(int)
train['Cabin_C'] = train['Cabin'].fillna('').str.count('C').astype(int)
train['Cabin_E'] = train['Cabin'].fillna('').str.count('E').astype(int)
train['Cabin_G'] = train['Cabin'].fillna('').str.count('G').astype(int)
train['Cabin_D'] = train['Cabin'].fillna('').str.count('D').astype(int)
train['Cabin_A'] = train['Cabin'].fillna('').str.count('A').astype(int)
train['Cabin_B'] = train['Cabin'].fillna('').str.count('B').astype(int)
train['Cabin_F'] = train['Cabin'].fillna('').str.count('F').astype(int)
train['Cabin_T'] = train['Cabin'].fillna('').str.count('T').astype(int)

train['Name_Miss'] = train['Name'].str.contains('Miss.').astype(int)
train['Name_Mrs'] = train['Name'].str.contains('Mrs.').astype(int)
train['Name_Master'] = train['Name'].str.contains('Master.').astype(int)
train['Name_Col'] = train['Name'].str.contains('Col.').astype(int)
train['Name_Major'] = train['Name'].str.contains('Major.').astype(int)
train['Name_Mr'] = train['Name'].str.contains('Mr.').astype(int)
train['Name_Dr'] = train['Name'].str.contains('Dr.').astype(int)
train['Name_Don'] = train['Name'].str.contains('Don.').astype(int)
train['Name_Sir'] = train['Name'].str.contains('Sir.').astype(int)

train['Ticket_cat'] = train['Ticket'].map(lambda x: list(train['Ticket'].unique()).index(x) if x in list(train['Ticket'].unique()) else 0)
train['Ticket_num'] = train['Ticket'].str.split().map(lambda x: x[1] if len(x) > 1 and x[1].isnumeric() else x[0] if x[0].isnumeric() else 0)
```
