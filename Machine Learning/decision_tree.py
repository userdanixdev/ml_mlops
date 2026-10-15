import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score, mean_squared_error,ConfusionMatrixDisplay, confusion_matrix,precision_score,f1_score,recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)


# Quantidade de alunos:
n = 10000

# Horas estudadas: entre 0 e 20 horas
horas_estudadas = np.random.uniform(0, 20, n)

# Score da prova: relacionado às horas estudadas
score_prova = (
    30
    + (horas_estudadas * 3)
    + np.random.normal(0, 10, n)
)

# Limita as notas entre 0 e 100
score_prova = np.clip(score_prova, 0, 100)

# Probabilidade de aprovação
prob_pass = 1 / (1 + np.exp(
    -(
        (horas_estudadas - 10) / 1.5
        + (score_prova - 50) / 15
    )
))

# Gera 0 ou 1 de acordo com a probabilidade
passou = np.random.binomial(1, prob_pass)

# DataFrame
df = pd.DataFrame({
    "horas_estudadas": horas_estudadas,
    "score_prova": score_prova,
    "pass": passou
})

print(df.head(10))

X = df[["horas_estudadas","score_prova"]]
y = df["pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state= 42)
print(f"Training Data: {X_train.shape},{y_train.shape}")
print(f"Testing Data: {X_test.shape},{y_train.shape}")

# Modelo DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42,max_depth=8)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accurary = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='weighted')
f1 = f1_score(y_test,y_pred,average='weighted')
recall = recall_score(y_test,y_pred,average='weighted')
print(f'Acurácia: {accurary}, Precisão: {precision}, Recall: {recall}, f1: {f1}\n')
print(f'Tree Depth: {model.get_depth()}\n')
print(f'Number of leaves: {model.get_n_leaves()}')

# Modelo RandomForest

model_2 = RandomForestClassifier(random_state=42, n_estimators=500, max_depth=20, max_leaf_nodes=12)
model_2.fit(X_train,y_train)

# mostrar no terminal:

tree_index = 0
tree_to_visualize = model_2.estimators_[tree_index]
plt.figure(figsize=(20,10))
plot_tree(
    tree_to_visualize,
    filled=True,
    feature_names=X.columns,
    class_names=["Reprovado", "Aprovado"],
    rounded=True,
    fontsize=9
)

plt.title("Árvore de Decisão - Random Forest")
plt.show()