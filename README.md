# Titanic Kaggle Competition
## About
The [Titanic challenge](https://www.kaggle.com/competitions/titanic/overview) on Kaggle is a competition in which the task is to predict the survival or the death of a given passenger based on a set of variables describing him such as his age, his sex, or his passenger class on the boat.

## Table of Contents
* [Titanic Kaggle Competition](#titanic-kaggle-competition)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#Installation)
  * [Data Preparation](#data-preparation)
  * [Hyperparameter Search](#hyperparameter-search)
  * [Model Evaluate](#model-evaluate)
    * [Logistic Regression](logistic_regression.ipynb)
    * [Neural Network](neural_network.ipynb)
    * [Random Forest](random_forest.ipynb)
    * [XGBoost](xgboost.ipynb)
  * [Error Analysis](#error-analysis)
  * [Final Model](#final-model)
  * [Entry](#entry)

## Installation
Cloning the repository and downloading the libraries used in the models
```Console
git clone https://github.com/guidasneves/titanic_kaggle.git
pip install requirements
```

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

Selecting only the feats that I will use in the model and also transforming null values ​​into 0.
```Python
variaveis = ['Sex_b', 'Age', 'Pclass', 'Embarked_S', 'Embarked_C', 'SibSp', 'Parch', 'Fare', 'Cabin_null',
             'Cabin_C', 'Cabin_E', 'Cabin_G', 'Cabin_D', 'Cabin_A', 'Cabin_B', 'Cabin_F', 'Cabin_T',
             'Name_Miss', 'Name_Mrs', 'Name_Master', 'Name_Col', 'Name_Major', 'Name_Mr', 'Name_Dr', 'Name_Don',
             'Name_Sir', 'Ticket_num', 'Ticket_cat']
             
X = train[variaveis].fillna(0)
X_test = test[variaveis].fillna(0)
y = train['Survived']
```

## Hyperparameter Search
Searching for the best hyperparameters using `Bayesian Optimization`.
```Python
def fit_model(hparams):
    n_estimators = hparams[0]
    max_depth = hparams[1]
    min_child_weight = hparams[2]
    subsample = hparams[3]
    learning_rate = hparams[4]
    num_parallel_tree = hparams[5]
    #colsample_bynode = hparams[6]
    
    model = XGBClassifier(n_estimators=n_estimators,
                          max_depth=max_depth,
                          min_child_weight=min_child_weight,
                          subsample=subsample,
                          learning_rate=learning_rate,
                          #colsample_bynode=colsample_bynode,
                          num_parallel_tree=num_parallel_tree,
                          n_jobs=-1,
                          random_state=42)
    
    model.fit(X_opt, y)
    
    yhat_train = model.predict(X_opt)
    
    return -accuracy_score(yhat_train, y)

space = [
    (100, 800), # n_estimators
    (10, 400), # max_depth
    (1, 50), # min_child_weight
    (0.07, 1.0), # subsample
    (1e-5, 1e-1, 'log-uniform'), # learning_rate
    (2, 50), # num_parallel_tree
    #(0.5, 1.0), # colsample_bynode
]

X_opt = StandardScaler().fit_transform(X)
opt = gp_minimize(fit_model, space, random_state=42, verbose=0, n_calls=50, n_random_starts=20)
opt.x
```

## Model Evaluate
Training and evaluating the model. I am using `RepeatedKFold` to divide the data between training and validation.

I will use `XGBoost` as an example, but I created and trained other models too.
* [Logistic Regression](logistic_regression.ipynb)
* [Neural Network](neural_network.ipynb)
* [Random Forest](random_forest.ipynb)
* [XGBoost](xgboost.ipynb)
```Python
kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)

scaler = StandardScaler()

step_train = []
step_cv = []

for linhas_train, linhas_cv in kf.split(X):
    X_train, X_cv = X.iloc[linhas_train].copy(), X.iloc[linhas_cv].copy()
    y_train, y_cv = y.iloc[linhas_train].copy(), y.iloc[linhas_cv].copy()

    X_train = scaler.fit_transform(X_train)
    X_cv = scaler.transform(X_cv)

    model = XGBClassifier(learning_rate=opt.x[4],
                         n_estimators=opt.x[0],
                         max_depth=opt.x[1],
                         min_child_weight=opt.x[2],
                         subsample=opt.x[3],
                         num_parallel_tree=opt.x[5],
                         n_jobs=-1, 
                         random_state=42)
    
    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    yhat_cv = model.predict(X_cv)

    acc_train = accuracy_score(yhat_train, y_train)
    acc_cv = accuracy_score(yhat_cv, y_cv)

    print(f'acc_train: {acc_train:.4f}, acc_cv: {acc_cv:.4f}\n')


    step_train.append(acc_train)
    step_cv.append(acc_cv)

print(f'Train mean: {np.mean(step_train):.2f}, CV mean: {np.mean(step_cv):.2f}')
```

## Error Analysis
Performing error analysis on incorrect generalizations in the cv set. Analyzing errors by gender.
```Python
X_cv_erro = train.iloc[linhas_cv].copy()
X_cv_erro['yhat'] = yhat_cv
X_cv_erro.head()

erro = X_cv_erro[X_cv_erro['Survived'] != X_cv_erro['yhat']]
erro = erro[['Survived', 'yhat', 'Name', 'Cabin', 'Embarked', 'Sex', 'Sex_b', 'Age', 'Pclass', 'Embarked_S',
             'Embarked_C', 'SibSp', 'Parch', 'Fare', 'Cabin_null',
             'Cabin_C', 'Cabin_E', 'Cabin_G', 'Cabin_D', 'Cabin_A', 'Cabin_B', 'Cabin_F', 'Cabin_T',
             'Name_Miss', 'Name_Mrs', 'Name_Master', 'Name_Col', 'Name_Major', 'Name_Mr', 'Name_Dr', 'Name_Don', 'Name_Sir']]

female = erro[erro['Sex_b'] == 1]
male = erro[erro['Sex_b'] == 0]

female.sort_values('Survived')
```

## Final Model
Retraining the final model with all the data.
```Python
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
model = XGBClassifier(learning_rate=1e-3,
                         n_estimators=opt.x[0],
                         max_depth=opt.x[1],
                         min_child_weight=opt.x[2],
                         subsample=opt.x[3],
                         num_parallel_tree=2,
                         n_jobs=-1, 
                         random_state=42)
    
model.fit(X, y)

yhat = model.predict(X_test)
yhat
```

## Entry
Creating the Set with the results of the predictions to import into kaggle.
```Python
result = pd.Series(yhat.reshape(-1), index=test['PassengerId'], name='Survived')
result

result.to_csv('./yhat/xgboost_model.csv', header=True)
```
