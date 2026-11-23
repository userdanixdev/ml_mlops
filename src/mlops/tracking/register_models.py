
import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlops.tracking.mlflow_config import configure_mlflow

load_dotenv()

REGISTERED_MODEL_NAME = "breast-cancer-classifier"

REGISTERED_RUNS = {
    "Random Forest Baseline": os.getenv("MLFLOW_RUN_RANDOM_FOREST"),
    "Logistic Regression Baseline": os.getenv("MLFLOW_RUN_LOGISTIC_REGRESSION"),
    "Random Forest RandomizedSearchCV": os.getenv("MLFLOW_RUN_RANDOMIZED_SEARCH"),
    "Random Forest HalvingGridSearchCV": os.getenv("MLFLOW_RUN_HALVING_SEARCH"),
}
# verificar se todos os Run IDs necessários foram carregados corretamente.
def validate_run_ids():
    missing_runs = [
        model_name
        for model_name, run_id in REGISTERED_RUNS.items()
        if not run_id
    ]

    if missing_runs:
        raise ValueError(
            "Run IDs ausentes no arquivo .env: "
            + ", ".join(missing_runs)
        )

# Criar uma versão no Registry
def register_model_version(client, model_name, run_id):

    existing_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    # Evitar duplicidade de registros:
# Aqui comparamos o Run ID que estamos tentando registrar com os Run IDs das versões existentes.    
    for version in existing_versions:
        if version.run_id == run_id:
            print(
                f"\nModelo já registrado: {model_name} "
                f"(versão {version.version})"
            )
            return
# O método search_logged_models() procura os modelos logados
#  cuja origem corresponde ao Run ID informado.
    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run_id}'",
        output_format="list",
    )
# O resultado é uma lista de modelos logados.
# Se a busca não retornar nenhum modelo, interrompemos o processo;
# evita tentar registrar um modelo inexistente.
    if not logged_models:
        raise ValueError(
            f"Nenhum modelo encontrado para o Run ID: {run_id}"
        )
# O next() retorna o primeiro elemento encontrado que satisfaça uma condição.
    logged_model = next(
        (
            model
            for model in logged_models
            if model.name == "sklearn_model"
        ),
        logged_models[0],
    )
# Construir o endereço do modelo
# Aqui construímos o identificador que permite ao MLflow localizar o modelo logado.
# models:/<model_id> permite referenciar diretamente um modelo logado.
    model_uri = f"models:/{logged_model.model_id}"
# Registrar no Model Registry:
# O método mlflow.register_model() cria uma versão do modelo no Registry.
    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        tags={
            "source_run_name": model_name,
            "validation_status": "pending", # Indica que ainda precisamos avaliar o modelo.
        },
    )

    print(f"\nModelo registrado: {model_name}")
    print(f"Versão: {registered_version.version}")
    print(f"Run ID: {run_id}")
    print(f"Logged Model ID: {logged_model.model_id}")
# Essas mensagens permitem acompanhar o resultado da operação no terminal.
# O MLflow retorna o objeto registered_version, que contém informações sobre a versão criada.    


def main():

    load_dotenv()
    validate_run_ids()
    configure_mlflow()
# Essa linha informa ao MLflow que o Registry utilizará o mesmo armazenamento configurado para o Tracking.
# Assim, Tracking e Registry utilizam o mesmo banco SQLite.
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
# Instancia o cliente que será utilizado para consultar as versões existentes.
    client = MlflowClient()

    print("\n========== MLflow MODEL REGISTRY ==========")
    print(f"Modelo registrado: {REGISTERED_MODEL_NAME}")
# Percorre os quatro modelos definidos no dicionário.
    for model_name, run_id in REGISTERED_RUNS.items():
        register_model_version(
            client=client,
            model_name=model_name,
            run_id=run_id,
        )

    print("\nTodos os modelos foram processados no Registry!")


if __name__ == "__main__":
    main()
