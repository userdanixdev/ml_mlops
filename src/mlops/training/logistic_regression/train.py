
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from utils.save_model import save_model
from utils.save_metrics import save_metrics
import pandas as pd


def train_model(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=421
        ))
    ])

    model.fit(X_train, y_train)
    save_model(model, "model_2_logistic_regression")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=["malignant", "benign"],
        zero_division=0,
        output_dict=True
    )

    save_metrics(
        model_name="Logistic Regression",
        accuracy=accuracy,
        classification_metrics=report_dict
    )

    print("\nAvaliação do modelo")
    print("*" * 40)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nMatriz de confusão:")
    print(matrix)

    print("\nRelatório de classificação:")
    print(pd.DataFrame(report_dict).T)
