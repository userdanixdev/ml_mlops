# Avaliação do modelo

## Metodologia

A avaliação utiliza os registros que possuem resultados conhecidos na base sintética.

Os dados foram divididos em:

| Conjunto | Quantidade |
|---|---:|
| Total de registros conhecidos | 8.000 |
| Treinamento (80%) | 6.400 |
| Teste (20%) | 1.600 |

A divisão foi realizada com `train_test_split`, utilizando `random_state=42` e estratificação das classes.

O modelo foi treinado no conjunto de treinamento e avaliado no conjunto de teste.

## Métricas obtidas

| Métrica | Resultado |
|---|---:|
| Acurácia (Accuracy) | 96,25% |
| Precisão (Precision) | 93,77% |
| Revocação (Recall) | 99,27% |
| F1-score | 96,44% |
| Loss final de treinamento | 0,2431 |

A perda apresentada corresponde ao valor final do treinamento, não à perda calculada exclusivamente no conjunto de teste.

## Matriz de confusão

A matriz de confusão obtida foi:

| | Previsto: Reprovado (0) | Previsto: Aprovado (1) |
|---|---:|---:|
| Real: Reprovado (0) | 727 | 54 |
| Real: Aprovado (1) | 6 | 813 |

Interpretação:

- **727 verdadeiros negativos:** estudantes classificados corretamente como reprovados.
- **54 falsos positivos:** estudantes reprovados classificados como aprovados.
- **6 falsos negativos:** estudantes aprovados classificados como reprovados.
- **813 verdadeiros positivos:** estudantes classificados corretamente como aprovados.

## Interpretação das métricas

### Acurácia — 96,25%

Representa a proporção total de classificações corretas no conjunto de teste.

### Precisão — 93,77%

Entre os registros classificados como aprovados, aproximadamente 93,77% eram realmente aprovados.

### Recall — 99,27%

O modelo identificou aproximadamente 99,27% dos registros realmente pertencentes à classe aprovado.

### F1-score — 96,44%

É a média harmônica entre precisão e recall, permitindo analisar o equilíbrio entre essas duas métricas.

## Conclusão

Neste experimento, o modelo apresentou alta acurácia e recall para a classe aprovado, identificando corretamente a maioria dos registros dessa classe.

A matriz de confusão também evidencia erros de classificação, reforçando a importância de analisar diferentes métricas em vez de considerar apenas a acurácia.

## Limitações

Os dados utilizados são sintéticos e foram gerados com características e regras definidas durante o desenvolvimento. Isso limita a generalização dos resultados para cenários reais.

Além disso, o modelo utilizado possui uma única camada linear, sendo uma abordagem de classificação simples, com comportamento semelhante ao de uma regressão logística.

Para aplicações reais, seria necessário utilizar dados representativos, avaliar possíveis vieses, validar a generalização e considerar os impactos de decisões automatizadas.