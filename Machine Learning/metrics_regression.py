import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error, classification_report, accuracy_score, mean_absolute_error, r2_score, mean_squared_error,ConfusionMatrixDisplay, confusion_matrix

# Gerar dataset automático:
# Square_Feet -> Área do imóvel em pés quadrados (ft²) -> Variável independente X
# House_Price -> Preço do Imóvel -> Variável dependente y
# Assim: Quanto maior a área do imóvel, maior tende a ser o preço.

np.random.seed(42)

# Quantidade de imóveis
n = 1000

# Area dos imóveis em m²:
square_meters = np.random.uniform(40,300,n)

# Preço base: aproxidamente R$ 5.000 por m²
noise = np.random.normal(0,5000,n)

house_price = (square_meters * 5000) + noise

# Criando DataFrame:
df = pd.DataFrame({
    "Square_Meters": square_meters,
    "House_Price":house_price
})
print(df.head())
print(df.shape)
# Visualization:

plt.scatter(
    df["Square_Meters"],
    df["House_Price"]
)
plt.xlabel("Área do imóvel (m²)")
plt.ylabel("Preço do imóvel (R$)")
plt.title("Preço do imóvel X área")
plt.show()

# Gerar o modelo:

X = df[["Square_Meters"]]
y = df["House_Price"]

X_train, X_test, y_train, y_test =  train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
# Display:
print(f"Intercept: {modelo.intercept_}")
print(f"Coeficiente: {modelo.coef_[0]}")
y_pred = modelo.predict(X_test)

print("Predição Preços:", y_pred)
print("Actual Prices:", y_test.values)

MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
r2 = r2_score(y_test, y_pred)

print(f"MAE: R$ {MAE:,.2f}")
print(f"MSE:  {MSE:,.2f}")
print(f"RMSE: R$ {RMSE:,.2f}")
print(f"R²:  {r2:,.4f}")