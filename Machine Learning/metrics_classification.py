# Métricas de Avaliação em Classificação:

# Ao treinar modelos de classificação, como regressão logística, é essencial avaliar o desempenho.
# Cada métrica mede um aspecto difererente do desempenho do modelo.

# 1. Acurácia ( Accuracy )
# 1.1 - Mede a proporção de previsões corretas em relação ao total de previsões.
# São boas para classes balanceadas. Pode ser enganosa para bases desbalanceadas.

# 2. Precisão ( Precision )

# Mede a proporção dos exemplos previstos como positivos que realmente são positivos.

# 3. Recall ( Sensibilidade/ Revocação )

# - Medição da proporção de exemplos positivos que foram corretamente identificados.

# 4. F1-Score:

# - Combina Precisão e Recall em uma única métrica ( média harmônica)

# 5. Curva ROC ( Receiver Operating Characteristic)
# - Avalia o desempenho do modelo para diferentes decisões. útil para comparar modelos em cenários de 
# classificação binária.

# Eixo X: Taxa de Falsos Positivos (FPR)
# Eixo Y: Taxa de Verdadeiros Positivos (TPR/Recall)
# Área da curva - Medição da qualidade geral:
# - AUC = 1 -> Modelo perfeito
# - AUC >= 0.5 -> Modelo aleatório

# Resumo das Métricas:
# Acurácia -> Percentual de Acertos -> Quando usar? Classes Balanceadas
# Precisão -> Confiabilidade dos positivos -> Quando usar? Evitar falsos positivos
# Recall -> Capacidade de encontrar positivos -> Quando usar? Evitar falsos negativos
# F1-Score -> Equilíbrio entre Precisão e Recall -> Quando usar? Classes Desbalanceadas
# ROC/AUC -> Desempenho em todos os limiares -> Quando usar? Comparação geral de modelos


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

# Divisão entre testes e alvo ( X e Y ):
X = df[['horas_estudadas']] # Features(s)
y = df['passou'] # Target variável ( 0 = Fail, 1 = Pass)
# Split data into 80% treino e 20% testes:
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# Display:
print(f"Training Data: {X_train.shape},{y_train.shape}")
print(f"Testing Data: {X_test.shape},{y_test.shape}")
print()
# inicializar o modelo de regressão
model = LogisticRegression()
# Treinar o modelo:
model.fit(X_train,y_train)
# Display:
print(f"Intercept: {model.intercept_}")
print(f"Coeficiente: {model.coef_[0]}")
# Fazer as predições com o dataset TEST:
y_pred = model.predict(X_test)
# Display:
print("Predicted Outcomes (Pass/Fail):", y_pred) # Os verdadeiros e falsos
print("Actual Outcomes:", y_test.values) # Os verdadeiros resultados
print("\nUtilizando Métricas de Avaliação:")
accuracy = accuracy_score(y_test,y_pred)
# Gerar a matriz de confusão para verificar o total de erros e acertos:
conf_matrix = classification_report(y_test,y_pred)
# Gerar a classificação:
class_report = classification_report(y_test,y_pred)
print(f"""
Acurácia: {accuracy}
Confusion Matrix:{conf_matrix}
Classification Report: {class_report}  """)


