"""
K-Nearest Neighbors (KNN) Classifier Example
A simple implementation demonstrating KNN on the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and train the KNN classifier
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"KNN Classifier (k={k})")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Example prediction
    sample = X_test[0].reshape(1, -1)
    prediction = knn.predict(sample)
    print(f"\nSample prediction:")
    print(f"  Features: {X_test[0]}")
    print(f"  Predicted class: {iris.target_names[prediction[0]]}")
    print(f"  Actual class: {iris.target_names[y_test[0]]}")


if __name__ == "__main__":
    main()
