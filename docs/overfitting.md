# Overfitting:

## Exemplo de Overfitting: Perda de Treino e Validação

![Overfitting](images/example_overfitting_dataset_moon.png)

- Linha azul - Loss de treino: continua diminuindo, indicando que o modelo reduz o erro nos dados utilizados para aprender.
- Linha laranja - Val Loss: diminui inicialmente, mas começa a aumentar depois de atingir seu menor valor.
- Linha tracejada - Época 12: marca aproximadamente o ponto a partir do qual o overfitting começa a ficar evidente neste exemplo ilustrativo.

O que acontece? O modelo continua aprendendo os dados de treinamento, mas seu desempenho nos dados de validação começa a piorar. Isso pode indicar que ele está se ajustando excessivamente aos exemplos de treinamento e perdendo capacidade de generalização.

**Importante**: os valores deste gráfico são ilustrativos, não foram extraídos do seu treinamento real. Nos gráficos que você enviou anteriormente, a perda de validação continuava diminuindo até a 20ª época; portanto, eles não mostram esse padrão de overfitting.