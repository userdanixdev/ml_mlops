## Preparação e Análise dos Dados

Utilizando o dataset Breast Cancer Wisconsin, disponibilizado pelo Scikit-learn o objetivo será preparar os dados para que possam ser utilizados no treinamento e na avaliação dos modelos de Machine Learning.
O que vamos desenvolver
1. Obtenção dos dados
   
   Carregar o Breast Cancer Wisconsin diretamente pelo Scikit-learn e compreender sua estrutura.

2. Análise exploratória (EDA)
   
   Investigar dimensões, tipos de dados, estatísticas descritivas e distribuição das classes.
3. Limpeza dos dados
   
   Verificar se possue valores ausentes, duplicidades e possíveis inconsistências.

4. Pré-processamento
   
   Separar variáveis preditoras e variável-alvo, dividir os dados em treino e teste e aplicar normalização quando apropriado.

5. Documentação
   
   Registrar as análises, decisões e resultados no projeto.


   
O que acontece com o target no pré-processamento?
O target não deve passar pela mesma normalização ou padronização das features.

Ele já está codificado numericamente como 0 e 1. Portanto:
- X: poderá receber padronização ou normalização.
- y: permanece com seus valores originais (0 e 1).
- train_test_split: divide ambos, mantendo a correspondência entre as amostras.
Uma observação: sua função check_transformations(X) verifica somente as features. Isso é intencional, pois o target é tratado separadamente.   