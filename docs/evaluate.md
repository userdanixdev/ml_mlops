# Avaliação do Modelo - Dataset Moon

O módulo `evaluate.py` é responsável por avaliar o desempenho da rede neural MLP (Multilayer Perceptron) utilizando os dados de teste. Essa etapa permite verificar a capacidade de generalização do modelo, analisando suas previsões em amostras que não participaram do treinamento.

A avaliação é realizada por meio de duas abordagens complementares:

- **Métricas de avaliação do Keras:** calculadas por `model.evaluate()`, incluindo a função de perda e as métricas definidas durante a compilação do modelo.
- **Relatório de classificação:** gerado por `classification_report()`, apresenta métricas detalhadas para cada classe prevista.

## Estrutura da função

A função `avaliar_modelo()` recebe o modelo treinado e os conjuntos de teste `X_test` e `y_test`.

```python
from sklearn.metrics import classification_report


# Avaliação final - Dataset Moon
def avaliar_modelo(model, X_test, y_test):

    # Avaliação com os dados de teste
    resultado = model.evaluate(
        X_test,
        y_test,
        verbose=0,
        return_dict=True
    )

    print("\nAvaliação final do modelo:")

    # Exibir todas as métricas da avaliação
    for metrica, valor in resultado.items():
        print(f"{metrica}: {valor:.4f}")

    # Gerar as probabilidades previstas pelo modelo
    y_pred_prob = model.predict(X_test, verbose=0)

    # Converter as probabilidades em classes 0 ou 1
    y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

    # Exibir o relatório de classificação
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Retornar as métricas da avaliação
    return resultado
```

## 3. Avaliação com `model.evaluate()`

O método `model.evaluate()` calcula a função de perda e as métricas configuradas em `model.compile()`, utilizando os dados de teste.

| Componente | Descrição |
|---|---|
| `model` | Rede neural treinada que será avaliada. |
| `X_test` | Características de entrada das amostras de teste, correspondentes às variáveis `x1` e `x2`. |
| `y_test` | Classes reais associadas às amostras de teste: `0` ou `1`. |
| `verbose=0` | Desativa a exibição do progresso da avaliação no terminal. |
| `return_dict=True` | Retorna as métricas em um dicionário, associando cada nome ao seu valor. |
| `resultado` | Armazena o dicionário com as métricas calculadas. |

### Exibição das métricas

O laço `for` percorre todas as métricas presentes no dicionário `resultado`.

```python
for metrica, valor in resultado.items():
    print(f"{metrica}: {valor:.4f}")
```

- `resultado.items()` permite percorrer os pares de chave e valor do dicionário.
- `metrica` representa o nome da métrica.
- `valor` representa o resultado numérico correspondente.
- `:.4f` formata o valor com quatro casas decimais.

As métricas exibidas dependem da configuração utilizada na compilação do modelo.

## Geração das previsões

Após calcular as métricas de avaliação, o modelo gera previsões para as amostras de teste.

```python
y_pred_prob = model.predict(X_test, verbose=0)
```

O método `model.predict()` retorna as probabilidades estimadas pela camada de saída. Como o modelo utiliza a função de ativação `sigmoid`, cada previsão representa um valor entre `0` e `1`.

### Conversão das probabilidades em classes

```python
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()
```

A expressão converte as probabilidades em classes binárias utilizando o limite de decisão `0.5`.

- Probabilidade menor que `0.5`: classe `0`.
- Probabilidade igual ou maior que `0.5`: classe `1`.
- `.astype(int)`: converte os resultados em números inteiros.
- `.ravel()`: transforma o resultado em um array unidimensional.

O limite de `0.5` é uma escolha inicial comum para classificação binária, podendo ser ajustado conforme os objetivos do problema.

## 5. Relatório de classificação

```python
print(classification_report(y_test, y_pred))
```

A função `classification_report()`, disponibilizada pelo Scikit-learn, compara as classes reais (`y_test`) com as classes previstas (`y_pred`) e apresenta as seguintes métricas:

| Métrica | Descrição |
|---|---|
| **Precision** | Proporção de previsões positivas corretas entre todas as previsões positivas realizadas. |
| **Recall** | Proporção de amostras positivas reais identificadas corretamente pelo modelo. |
| **F1-score** | Média harmônica entre precision e recall, equilibrando as duas métricas. |
| **Support** | Quantidade de amostras reais pertencentes a cada classe. |

O relatório também apresenta a acurácia geral e médias das métricas para o conjunto avaliado.

## Retorno da função

```python
return resultado
```

O retorno permite reutilizar as métricas em outras partes do projeto, como comparações entre arquiteturas MLP, análises de regularização e geração de relatórios.

Por exemplo, no arquivo `main.py`:

```python
resultado = avaliar_modelo(model, X_test, y_test)
```

A variável `resultado` recebe o dicionário de métricas calculadas por `model.evaluate()`.

O módulo `evaluate.py` é executado após o treinamento da rede neural. Seu papel é consolidar os resultados da avaliação e permitir uma análise mais detalhada do desempenho do modelo.

**Observação:** durante a comparação de arquiteturas e técnicas de regularização, utilize os resultados de validação para orientar a escolha do modelo. Reserve os dados de teste para a avaliação final do modelo selecionado, evitando utilizá-los repetidamente para tomar decisões de ajuste.



