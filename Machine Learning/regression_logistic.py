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

np.random.seed(42)

# Quantidade de alunos:
n = 10000

# Horas estudadas: entre 0 e 20 horas
horas_estudadas = np.random.uniform(0,20,n)
# Probabilidade de aprovação:
prob_pass = 1 / (1 + np.exp(-(horas_estudadas - 10)/ 1.5))
# Gera 0 ou 1 de acordo com a probabilidade:
passou = np.random.binomial(1,prob_pass)
# DataFrame:
df = pd.DataFrame({
    "horas_estudadas": horas_estudadas,
    "passou": passou
})

print(df.head())
print(df.shape)
print(df['passou'].value_counts())
print()
print("Regressão Logística\n")
X = df[['horas_estudadas']]
y = df['passou']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size = 0.2,random_state=42,stratify=y
)
mode_regression_logistic = LogisticRegression()
mode_regression_logistic.fit(X_train, y_train)
previsoes = mode_regression_logistic.predict(X_test)
print(f" Training Data: {X_train.shape},{y_train.shape}")
print(f" Testing Data: {X_test.shape},{y_train.shape}\n")
print("Acurácia:", accuracy_score(y_test, previsoes))
print(classification_report(y_test, previsoes))
print()
print("Predições\n")
y_pred = mode_regression_logistic.predict(X_test)
print("Predições: ( PASS/FAIL)", y_pred)
print("Atual Outcomes:", y_test.values)
print(" Criação de um gráfico de alcance de horas")
study_hours = np.linspace(X.min(),X.max(),100)
print("Calculo de probabilidade de predição usando a função sigmoid:\n")
y_prob = mode_regression_logistic.predict_proba(study_hours.reshape(-1,1))[:,1]
print("Plotar os pontos de dados\n")
plt.scatter(study_hours,y_prob, color='red',label='logistic Regression Curve')
# Adicionar labels and title
plt.xlabel("Study Hours")
plt.ylabel("Probabilidade de passar")
plt.title("Regressão Logística vs Pass")
plt.legend()
plt.show()


