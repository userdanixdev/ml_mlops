# Vértice Seguros - Estudos de Regressão com RNA 

## Objetivo do estudo

Este estudo tem como objetivo explorar as etapas de preparação, transformação e análise de dados aplicadas a um problema de regressão utilizando Python e PyTorch com Redes Neurais Aritificais **(RNA)**. O estudo busca compreender como características dos segurados podem ser utilizadas para estimar valores de seguros por meio de um modelo de aprendizado de máquina.

## Cenário proposto

A Vértice Seguros é uma empresa fictícia utilizada como cenário para este estudo. O dataset sintético contém 100 registros de segurados, com identificadores, idades entre 18 e 80 anos e valores simulados de seguros.

Para representar um cenário de previsão, 15 valores de seguros são ocultados intencionalmente. Esses registros poderão ser utilizados posteriormente para testar as previsões do modelo treinado com os dados conhecidos.

O projeto também explora a padronização e a normalização dos dados, permitindo comparar as transformações e observar seus efeitos antes da implementação do modelo de regressão.

## Geração do dataset

A geração dos dados utiliza NumPy e Pandas, permitindo controlar a aleatoriedade, simular variações nos valores dos seguros e preparar registros com valores ausentes.

### Principais recursos utilizados

- **`np.random.seed(42)`**: fixa a semente aleatória para reproduzir os mesmos dados em novas execuções, desde que a sequência de operações permaneça igual.
- **`np.random.randint(18, 81, size=N)`**: gera 100 idades inteiras entre 18 e 80 anos, incluindo ambos os limites.
- **`np.random.normal(0, 250, size=N)`**: gera variações aleatórias com média zero e desvio-padrão de 250, simulando diferenças nos valores dos seguros.
- **`efeito_idade`**: representa o aumento do valor sintético do seguro conforme a idade se distancia de 40 anos, utilizando uma relação quadrática.
- **`np.random.choice(...)`**: seleciona 15 índices distintos para ocultar os valores dos seguros.
- **`np.nan`**: representa valores ausentes no Pandas.
- **`to_csv(...)`**: salva o dataset original em CSV, preservando as idades na escala original, sem padronização ou normalização.

## Exemplo do script

```python
np.random.seed(42)

N = 100

idades = np.random.randint(18, 81, size=N)

efeito_idade = (idades - 40) ** 2

ruido = np.random.normal(0, 250, size=N)

valores_seguro = 1200 + 1.5 * efeito_idade + ruido
valores_seguro = np.maximum(valores_seguro, 500)

df = pd.DataFrame({
    "nome": [f"Segurado_{i:03d}" for i in range(1, N + 1)],
    "idade": idades,
    "valor_seguro": np.round(valores_seguro, 2)
})

indices_nulos = np.random.choice(
    df.index,
    size=15,
    replace=False
)

df.loc[indices_nulos, "valor_seguro"] = np.nan

df.to_csv(
    "src/dataset_seguros_vertice.csv",
    index=False
)
```

> **Nota:** A relação entre idade e valor do seguro foi definida artificialmente para permitir o estudo das etapas de um fluxo de Machine Learning.