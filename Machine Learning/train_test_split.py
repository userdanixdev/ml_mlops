import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score, mean_squared_error,ConfusionMatrixDisplay, confusion_matrix
from padronization_normalization import preparar_dados

# Exemplo de um DataFrame para divisão:

dados_normalizados = preparar_dados()

df_split = pd.DataFrame(dados_normalizados,columns=[f'feature_{i}' for i in range(1,11)])
print("\nDataFrame para divisão:")
print(df_split.head())
print("\nDimensões:") 
print(df_split.shape)

np.random.seed(42)

target = np.array(
    [0] * 7000 +
    [1] * 3000
)

np.random.shuffle(target)
df_split['target'] = target
print("\nDataFrame com target:") 
print(df_split.head())
print("\nDistribuição das classes:") 
print(df_split["target"].value_counts())

# Separação X e y:
X = df_split.drop(columns="target")  ## X para features
y = df_split['target'] 

print("\nDimensão de X:")
print(X.shape)
print("\nDimensão de y:")
print(y.shape)

# Divide 80% para treino e 20% para teste:
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(f"Tamanho do conjunto X_train: {X_train.shape}")
print(f"Tamanho do conjunto X_test: {X_test.shape}")
print(f" Proporção de classes no treino vs teste:\n")
print(pd.Series(y_train).value_counts(normalize=True).rename("train"))
print(pd.Series(y_test).value_counts(normalize=True).rename("test"))

# Initialized the Linear Regression model:
model_regression = LogisticRegression()
# Train the model on the training data:
model_regression.fit(X_train,y_train)
# Display the learned coefficients:
print(f"Intercept:{model_regression.intercept_}")
print(f"Coefficient:{model_regression.coef_[0]}")
print()
print("...Fazendo Previsões.." )
y_pred = model_regression.predict(X_test)
# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)
# Display das predições:
print("Classificação :", classification_report(y_test,y_pred))
print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")

# Calculate Mean Squared Error:
mse = mean_squared_error(y_test,y_pred)
# Calculo do R-squared:
r2 = r2_score(y_test,y_pred)
# Display the evaluation metrics:
print(f"Mean Squared Error: {mse}")
print(f" R-Squared: {r2}")

# Plot the data points:
# Visualização
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Classe 0", "Classe 1"]
)

disp.plot()

plt.title("Matriz de Confusão - Classificação")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.show()

