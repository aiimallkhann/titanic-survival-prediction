# Titanic Survival Prediction

Built this because I wanted to get my hands dirty.

This is my first ML project. I predicted whether a Titanic passenger survived or not based on their data — age, sex, ticket class, cabin deck, and the title extracted from their name. No step-by-step tutorial. Just me figuring out what the next line of code was going to be, which was honestly the hardest part.

## What I used

- Random Forest Classifier
- Features: Pclass, Sex, Age, SibSp, Parch, Title (one-hot encoded), Deck (one-hot encoded)
- `max_depth=4` made the biggest difference — without it the model memorized the training data instead of actually learning

## Results

- Validation accuracy: 0.81
- Kaggle score: 0.78

## How to run

1. Get `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data)
2. Drop them in the same folder as the script
3. Run it — prints validation accuracy and saves `submission.csv`

## Stack

Python, pandas, scikit-learn

---

*Aimal Khan — CS student working toward ML engineering*
