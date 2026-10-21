# Titanic Survival Prediction using Decision Tree Classifier

## Project Overview
This machine-learning project predicts whether a Titanic passenger survived using a Decision Tree Classifier and selected passenger details.

## Dataset
`Titanic-Dataset.csv` contains 891 rows and includes passenger information such as survival status, passenger class, gender, age, family aboard, fare, and embarkation port.

## Features Used
- `Pclass` — passenger class
- `Gender` — passenger gender
- `Age` — passenger age
- `SibSp` — siblings/spouses aboard
- `Parch` — parents/children aboard
- `Fare` — ticket fare
- `Embarked` — port of embarkation

Target: `Survived`

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn

## Project Files
- `main.py` — preprocessing, training, and evaluation code
- `Titanic-Dataset.csv` — dataset
- `requirements.txt` — required Python packages
- `.gitignore` — excludes local environment and cache files

## How to Run
1. Install Python.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the program:

   ```bash
   python main.py
   ```

## Evaluation
The script prints model accuracy and a classification report containing precision, recall, and F1-score.

## Note
This is an educational machine-learning project. The model's predictions are estimates based on the supplied dataset.
