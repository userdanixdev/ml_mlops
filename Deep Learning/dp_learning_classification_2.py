import numpy as np
import pandas as pd

np.random.seed(42)

N = 10_000
percent_sem_result = 0.20

# Lista de Nomes:

nomes = [
    'Ana',
    'Bruno',
    'Carlos',
    'Daniel',
    'Eduardo',
    'Fernanda',
    'Gabriel',
    'Helena',
    'Igor',
    'Juliana',
    'Lucas',
    'Mariana',
    'Rafael',
    'Camila',
    'Pedro',
    'Larissa',
    'Gustavo',
    'Beatriz',
    'Felipe',
    'Amanda'
]

sobrenomes = [
    'Silva',
    'Santos',
    'Oliveira',
    'Souza',
    'Lima',
    'Pereira',
    'Costa',
    'Almeida',
    'Ferreira',
    'Rodrigues',
    'Martins',
    'Carvalho',
    'Gomes',
    'Ribeiro',
    'Alves',
    'Moura',
    'Barbosa',
    'Nascimento',
    'Teixeira',
    'Araújo'
]

# Concatenando os nomes:

first_name = np.random.choice(
    nomes,
    size=N
)
last_name = np.random.choice(
    sobrenomes,
    size = N
)
complete_names = np.char.add(
    np.char.add(first_name,' '),
    last_name
)

# Geração das features:

estudou = np.random.uniform(
    0,1,N
)
conhecimento_previo = np.random.uniform(
    0,1,N
)
facilidade_aprendizado = np.random.uniform(
    0,1,N
)

# Gerar Scores para os resultados:

score = (
    0.5 * estudou
    + 0.3 * conhecimento_previo
    + 0.2 * facilidade_aprendizado
)

# Gerando resultado:

resultado = (
    score >= 0.5
).astype(float)

# Criando DataFrame:

df = pd.DataFrame({
    'nome':complete_names,
    'estudou': estudou,
    'conhecimento_previo': conhecimento_previo,
    'facilidade_aprendizado': facilidade_aprendizado,
    'resultado': resultado
})

# Deixando parte dos resultados vazios:

quant_without_result = int(
    N * percent_sem_result
)
index_without_result = np.random.choice(
    N,size=quant_without_result,
    replace = False
)
df.loc[index_without_result,
       'resultado'] = np.nan

# Salvando o arquivo original:

df.to_excel(
    'alunos_base_original.xlsx',
    index = False
)

# Informações da base:

print("Base criada com sucesso!")
print()
print(f'Total de Alunos: {len(df)}')
print(
    f' Resultados preenchidos: '
    f'{df['resultado'].notna().sum()}'
)
print(
    f'Resultados vazios: '
    f'{df['resultado'].isna().sum()}'
)
print()
print('Primeirtos registros:\n')
print(df.head())
print()
print('Arquivo criado: alunos_base_original.xlsx')

# O script faz criar 10.000 alunos Usamos a biblioteca NumPy para gerar as características.
# Cada aluno terá três características.
# A criação do Score simula uma regra para gerar os pesos no resultado.
# O peso maior é pra quem estudou: 50%, conhecimento prévio = 30% e facilidade de aprendizado = 20%
# Assim a regra do jogo é quem estuda mais tem maior influência no resultado.
# Do total de dados preenchidos temos 8.000 com resultado conhecido e 2.000 com resultado vazio.
# Esses 2.000 alunos serão justamente aqueles que o nosso modelo deverá prever posteriormente.
