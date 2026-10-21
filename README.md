## EDA: Análise exploratória de dados

### O que é o Dataset Moon?

O ```make_moons`` é uma função do módulo ```sklearn.datasets``` da biblioteca **Scikit-learn.** 
Com essa função é possível gerar conjunto de dados sintéticos para estudar problemas de classificação binária não linear especialmente quando as classes possuem uma fronteira de decisão não linear.

O nome Moon vem do formato dos dados: duas regiões curvas que lembram duas meias-luas entrelaçadas. Cada ponto pertence a uma de duas classes. Diferentemente de um dataset real, os dados são gerados artificialmente.

A documentação oficial do make_moons(), do Scikit-learn, descreve a função como um gerador de duas classes de dados sintéticos em formato de duas meias-luas entrelaçadas, usado para testar algoritmos de classificação não linear.

*A documentação define X como uma matriz NumPy em formato (n_samples, 2), contendo as coordenadas de cada amostra. Assim também os vetores em NumPy em formato 0 e 1, contendo o rótulo de cada amostra.*