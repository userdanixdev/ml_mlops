from sklearn.model_selection import train_test_split
from pathlib import Path

def separate_features_target(data):
    X = data.drop(columns=["target"])
    y = data["target"]

    return X,y

def check_transformations(X):
    print("\nVerificação para pré-processamento")
    print("-" * 40)

    print(f"Valores ausentes: {X.isnull().sum().sum()}")
    print(f"Tipos numéricos: {X.select_dtypes(include='number').shape[1]}")
    print(f"Quantidade de features: {X.shape[1]}")

    print("\nEscalas das variáveis:")
    print(X.describe().loc[["min", "max"]].T)


def train_test_split_data(X,y,test_size=0.2, random_state=42):
    return train_test_split(X,y,test_size=test_size, random_state=random_state,stratify=y)

def generate_preprocessing_report(X,y,X_train,X_test,y_train,Y_test)    :
    report_path = Path("reports/preprocessing_report.md")
    report_path.parent.mkdir(parents=True,exist_ok=True)
    
