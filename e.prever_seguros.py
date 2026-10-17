from dp_learning_regression_1 import X_prever, df_prever
from treinar_modelo_2 import model
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Gerar previsões para os seguros ausentes
model.eval()

with torch.no_grad():
    previsoes_normalizadas = model(X_prever)

previsoes_numpy = previsoes_normalizadas.cpu().numpy().ravel()

print("\nPrevisões normalizadas:")
print(previsoes_numpy)

# Carregar dados originais
df_original = pd.read_csv("src/dataset_seguros_vertice.csv")

# Recuperar os valores de seguro conhecidos
seguros_conhecidos = (
    df_original["valor_seguro"]
    .dropna()
    .to_numpy()
    .reshape(-1, 1)
)

# Ajustar o scaler com os valores originais conhecidos
minmax_y = MinMaxScaler()
minmax_y.fit(seguros_conhecidos)

# Converter previsões normalizadas para valores monetários
previsoes_reais = minmax_y.inverse_transform(
    previsoes_numpy.reshape(-1, 1)
).ravel()

# Criar DataFrame com as previsões
df_resultados = df_prever[["nome", "idade"]].copy()

df_resultados["seguro_normalizado_previsto"] = previsoes_numpy
df_resultados["valor_seguro_previsto"] = previsoes_reais

# Exibir resultados
print("\nPrevisões na escala monetária original:")
print(df_resultados.to_string(index=False))

# Salvar resultados sem alterar os arquivos originais
pasta_saida = Path("analytics/predicoes")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivo_saida = pasta_saida / "previsoes_seguros.csv"

df_resultados.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nArquivo salvo em: {arquivo_saida.resolve()}")


# O que esse código faz?
# Recupera os valores de seguro conhecidos no dataset original.
# Ajusta um MinMaxScaler para reproduzir a transformação entre o valor monetário e a escala normalizada.
# Converte as 15 previsões para a escala monetária original.
# Mantém nome e idade associados a cada previsão.
# Salva os resultados em analytics/predicoes/previsoes_seguros.csv.