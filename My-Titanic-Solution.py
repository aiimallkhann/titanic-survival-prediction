# By Aimal Khan
# Titanic Survival Prediction — RandomForestClassifier
# Kaggle score: 0.78

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data = pd.get_dummies(data, columns=['Title'])
titleCols = [col for col in data.columns if col.startswith('Title_')]

data['Deck'] = data['Cabin'].str[0]
data = pd.get_dummies(data, columns=['Deck'])
deckCols = [col for col in data.columns if col.startswith('Deck_')]

y = data['Survived']
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'] + titleCols + deckCols
X = data[features].copy()
X['Age'] = X['Age'].fillna(data['Age'].median())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

titanicModel = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0)
titanicModel.fit(X_train, y_train)

validPredictions = titanicModel.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, validPredictions))

titanicModel.fit(X, y)

test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test = pd.get_dummies(test, columns=['Title'])
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
test['Deck'] = test['Cabin'].str[0]
test = pd.get_dummies(test, columns=['Deck'])
title_cols = [col for col in test.columns if col.startswith('Title_')]
deck_cols = [col for col in test.columns if col.startswith('Deck_')]
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'] + title_cols + deck_cols
testX = test[features].copy()
testX['Age'] = testX['Age'].fillna(testX['Age'].median())
testX = testX.reindex(columns=X.columns, fill_value=0)

predictions = titanicModel.predict(testX)
print(predictions)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)