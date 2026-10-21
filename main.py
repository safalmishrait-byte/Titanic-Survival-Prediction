"""
Titanic Survival Prediction using a Decision Tree Classifier

This script loads the local Titanic CSV file, preprocesses selected features,
trains a Decision Tree Classifier, and prints evaluation metrics.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_and_preprocess_data():
    data_path = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
    df = pd.read_csv(data_path)

    # Match the column names in the supplied CSV.
    features = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    data = df[features + [target]].copy()
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Gender"] = data["Gender"].map({"male": 0, "female": 1})

    # Convert categorical embarkation values into numeric columns.
    data = pd.get_dummies(data, columns=["Embarked"], drop_first=True, dtype=int)

    X = data.drop(columns=[target])
    y = data[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def main():
    print("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
