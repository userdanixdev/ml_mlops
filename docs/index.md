### O que é o Dataset Moon?

O ```make_moons``` é uma função do módulo **```sklearn.datasets```** da biblioteca **Scikit-learn.** 
Com essa função é possível gerar conjunto de dados sintéticos para estudar problemas de classificação binária não linear especialmente quando as classes possuem uma fronteira de decisão não linear.

O nome 'Moon' vem do formato dos dados: duas regiões curvas que lembram duas meias-luas entrelaçadas. Cada ponto pertence a uma de duas classes. Diferentemente de um dataset real, os dados são gerados artificialmente.

A documentação oficial do *make_moons()*, do **Scikit-learn**, descreve a função como um gerador de duas classes de dados sintéticos em formato de duas meias-luas entrelaçadas, usado para testar algoritmos de classificação não linear.


![dataset](images/scatter_dataset_moon.png)

A documentação define **X** como uma matriz NumPy em formato (n_samples, 2), contendo as coordenadas de cada amostra. Assim também os vetores em NumPy em formato 0 e 1, contendo o rótulo de cada amostra sendo nomeado como **'y'**

Para compreender a estrutura de X, utilizei seus atributos para identificar a quantidade de características disponíveis e criar nomes para as colunas:

```python
colunas = []
for i in range(X.shape[1]):
    nome_coluna = f"x{i+1}"
    colunas.append(nome_coluna)
print(colunas)   
```
Nesse trecho, X.shape[1] retorna a quantidade de colunas da matriz. Como o Dataset Moon possui duas características, o resultado é: **["x1","x2"]**

> Esses nomes foram definidos no código para facilitar a identificação das características durante a análise; eles não são nomes fornecidos originalmente pela função make_moons().

Além disso, utilizei a biblioteca Pandas para transformar os dados em um DataFrame, facilitando a inspeção de sua estrutura, a identificação das características e a análise dos rótulos das classes.

![eda](images/eda_2_moon.png)
![eda_1](images/EDA_moon.png)

> **Obs**: Caso queira reproduzir a exploração de dados local basta executar o script 'dataset_moon.py' que se encontra no repositório do GitHub. 

## Por que esse dataset é interessante para Machine Learning?

Um modelo linear, como a Regressão Logística sem transformações de características, normalmente não consegue separar perfeitamente as duas meias-luas com uma única linha reta.

Já modelos como K-Nearest Neighbors (KNN) ou árvores de decisão podem aprender fronteiras curvas. Isso nos permite estudar:

- Underfitting e overfitting.
- Separação entre treino e teste.
- Comparação entre modelos lineares e não lineares.
- Fronteiras de decisão e regiões de classificação.
- Métricas como precisão, recall, F1-score e matriz de confusão.








