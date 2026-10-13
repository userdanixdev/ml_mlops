import pandas as pd
import numpy as np

# Abrir o arquivo Excel com as previsões:
df = pd.read_excel('alunos_com_predicoes.xlsx')

print('='*50)
print('Verificações do arquivo de previsões')
print('='*50)

# Verificações de quantidade e colunas registradas:
print('\nDimensões do DataFrame: ')
print(f'Linhas: {df.shape[0]}')
print(f'Colunas: {df.shape[1]}')

# Verificar valores ausentes por coluna:
print('\n Valores ausentes por coluna: ')
print(df.isna().sum())

# Verificar resultados previstos:
print('\n Distribuião dos resultados previstos: ')
print(df['resultado_predito'].value_counts(dropna=False).sort_index())

# Verificação de quantos alunos receberam previsões:
mask = df['resultado'].isna()

print('\nResumo das previsões: ')
print(f' Alunos originalmente sem resultado: {mask.sum()}')
print('Alunos com previsão preenchida:',
      df.loc[mask,'resultado_predito'].notna().sum())
print('Alunos sem previsão: ',
      df.loc[mask,'resultado_predito'].isna().sum())

# Verificação de valores válidos na coluna previsão:
valores_validos = df['resultado_predito'].dropna().isin([0,1])
print('\n Validação dos resultados: ')
print('Previsões limitadas a 0 e 1:',
      valores_validos.all())
print('Quantidade de previsões inválidas: ',
      (~valores_validos).sum())
# Verificar probabilidades
probabilidades = df['probabilidade_predicao'].dropna()

print('\nValidação das probabilidades:')
print(
    'Todas estão entre 0 e 1:',
    probabilidades.between(0, 1).all()
)
print('Quantidade de probabilidades:', len(probabilidades))

# Exibir amostra das previsões
print('\nPrimeiros 10 alunos que receberam previsões:')
print(
    df.loc[
        mask,
        [
            'nome',
            'estudou',
            'conhecimento_previo',
            'facilidade_aprendizado',
            'resultado_predito',
            'probabilidade_predicao'
        ]
    ].head(10).to_string(index=False)
)

print('\nVerificação concluída!')      
