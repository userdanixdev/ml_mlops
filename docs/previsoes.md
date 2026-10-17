# Previsões:

A diferença entre as curvas será o valor constante de **bias_saida[0]**. A curva com bias representa a saída completa da rede, enquanto a curva sem bias mostra apenas a soma das contribuições dos dois perceptrons ocultos.

Como podemos verificar, a diferença é pequena:

![Perceptrons_bias](images/soma_perceptrons_com_sem_bias.png)

Os pesos controlam como as entradas e ativações contribuem para a previsão. O bias oferece um ajuste adicional que permite posicionar melhor as transformações e a saída. No modelo, os dois trabalham em conjunto para estimar o valor normalizado do seguro com menor erro.

## Por que o bias é aprendido durante o treinamento?

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
* **loss.backward()** calcula os gradientes dos parâmetros, incluindo os biases.

* **optimizer.step()** atualiza os parâmetros para tentar reduzir a função de perda.

> Observação importante: Portanto, no gráfico, ao comparar a soma com e sem bias_saida, está sendo isolado apenas a influência do bias da camada de saída. Os biases da camada oculta continuam presentes nas ativações dos dois perceptrons.

*O bias oferece um parâmetro adicional para o modelo aprender a relação entre idade e valor do seguro. Isso pode melhorar o ajuste, mas não garante uma previsão melhor por si só: o resultado depende dos dados, do treinamento e da arquitetura.*

![idadeXvalor_previsto](images/idade_vs_valor_previsto.png)