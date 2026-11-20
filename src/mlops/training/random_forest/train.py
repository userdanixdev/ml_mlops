from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from utils.save_model import save_model

def train_model(X_train,y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train,y_train)
    save_model(model)
    return model

def evaluate_model(model,X_test,y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["malignant","benign"],
        zero_division=0
    )
    print("\nAvaliação do modelo")
    print("*"*40)
    print(f"Accuracy:{accuracy:.4f}")

    print("\nMatriz de confusão:")
    print(matrix)

    print("\nRelatório de classificação:")
    print(report)

# - train_model(): recebe os dados de treinamento, instancia o Random Forest, executa o fit() e devolve o modelo treinado.
# - evaluate_model(): recebe o modelo treinado e os dados de teste, realiza previsões e calcula as métricas.
# Não precisamos de StandardScaler nesse primeiro modelo, pois Random Forest não depende da escala das variáveis.    