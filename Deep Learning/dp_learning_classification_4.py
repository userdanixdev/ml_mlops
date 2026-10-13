# Nova base de dados com as previsões:
import numpy as np
import dp_learning_classification_2 as step2
import dp_learning_classification_3 as step3

# Cópia do DataFrame original: 
df_resultado = step2.df.copy()

# Recuperação das variáveis 'previsões' e 'probabilidades':
previsoes = step3.previsoes
probabilidades = step3.probabilidades

# Cria a coluna de previsão usando o resultado original:
df_resultado['resultado_predito'] = df_resultado['resultado']

# identifica os alunos que não tinham resultado:
mask = df_resultado['resultado'].isna()

# Preenche a coluna de resultados previstos:
df_resultado['resultado_predito'] = df_resultado['resultado']

# Substitui os NaN pelas previsões do modelo:
df_resultado.loc[mask,'resultado_predito'] = (
    previsoes.squeeze().numpy()

)
# Cria a coluna de probabilidade:
df_resultado['probabilidade_predicao'] = (
    df_resultado['resultado'].astype(float)
)

# Coloca a probabilidade somente nos alunos que foram previstos:
df_resultado.loc[mask, 'probabilidade_predicao'] = (
    probabilidades.squeeze().numpy()
)

# Salva um novo arquivo:
df_resultado.to_excel(
    'alunos_com_predicoes.xlsx',
    index = False
)
print('Arquivo criado com sucesso!')
print('Total de registros:', len(df_resultado))
print('Valores NaN em resultado_predito:',
      df_resultado['resultado_predito'].isna().sum())