# Treinamento do Modelo

```python
hist = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=0)
```

O método `model.fit()` do Keras é utilizado para treinar a rede neural MLP (*Multilayer Perceptron*). O treinamento permite que o modelo ajuste seus pesos e aprenda os padrões presentes nos dados, relacionando as características de entrada (`X_train`) às classes correspondentes (`y_train`).

O parâmetro `validation_split=0.2` reserva 20% dos dados de treinamento para validação, permitindo acompanhar o desempenho do modelo durante o aprendizado, sem utilizar esses registros para atualizar os pesos. Como o conjunto de treinamento possui 800 registros, são utilizados 640 para o treinamento e 160 para validação.

![Treinamento](images/treinamento_moon.png)

Esse parâmetro divide os 800 registros de treinamento. 

* 640 registros: utilizados para atualizar os pesos da rede.
* 160 registros: utilizados para validação durante as épocas.

**Observação:** os 200 registros separados anteriormente para teste (`X_test` e `y_test`) permanecem isolados durante essa etapa. Eles serão utilizados posteriormente para avaliar o desempenho do modelo em dados que não participaram do treinamento nem da validação.


O parâmetro `epochs=20` define a quantidade de épocas, ou seja, o número de passagens completas pelos dados utilizados no treinamento. 

Já `batch_size=32` determina que os exemplos sejam processados em lotes de até 32 registros antes de cada atualização dos pesos. Como temos 800 registros der treino e usamos ```validation_split=0.2```, ficam 640 registros para treinar e 160 para validar. Dessa forma são 640 registros por 32 lotes que ficam com 20 lotes por época. Cada lote de 20 unidades repesentam 32 registros.

Nesse cenário, são realizadas 20 atualizações por época, totalizando até 400 atualizações ao longo do treinamento.

Por fim, `verbose=0` desativa a exibição do progresso no terminal, enquanto a variável `hist` armazena o histórico do treinamento, incluindo as métricas de desempenho calculadas em cada época, tanto para os dados de treinamento quanto para os de validação.

# Componentes do treinamento:

### Explicação de cada componente

| Componente | Explicação |
|---|---|
| `loss_test` | Armazena o valor da função de perda calculada durante a avaliação do modelo com os dados de teste. |
| `accuracy_test` | Armazena a acurácia do modelo nos dados de teste, indicando a proporção de previsões corretas. |
| `model.predict()` | Método do Keras que gera as previsões do modelo para os dados de entrada fornecidos. |
| `X_test` | Características do conjunto de teste, utilizadas para verificar como o modelo se comporta em dados que não participaram do treinamento. |
| `verbose=0` | Desativa a exibição do progresso durante a geração das previsões. |

