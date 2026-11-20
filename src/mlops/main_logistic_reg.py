from mlops.data.load_dataset import load_dataset

from mlops.preprocessing.pre_process import (
    separate_features_target,
    check_transformations,
    train_test_split_data,
    generate_preprocessing_report
)

from mlops.training.logistic_regression.train import (
    train_model,
    evaluate_model
)

def main():
    data = load_dataset()
    X, y = separate_features_target(data)
    check_transformations(X)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    generate_preprocessing_report(
        X, y, X_train, X_test, y_train, y_test
    )
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()