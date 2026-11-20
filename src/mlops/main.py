
from mlops.data.load_dataset import load_dataset
from mlops.preprocessing.pre_process import (
    separate_features_target,
    check_transformations,
    train_test_split_data,
    generate_preprocessing_report
)

from mlops.training.train import (
    train_model,
    evaluate_model
)


def main():
    # Carregamento
    data = load_dataset()

    # Separação entre features e target
    X, y = separate_features_target(data)

    # Verificação para pré-processamento
    check_transformations(X)

    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    generate_preprocessing_report(
    X, y, X_train, X_test, y_train, y_test
)

    print("\nDimensões dos conjuntos")
    print("-" * 40)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # treinamento
    model = train_model(X_train,y_train)
    # Avaliação
    evaluate_model(model,X_test,y_test)


if __name__ == "__main__":
    main()
