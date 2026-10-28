# Avaliação final - Dataset Moon
from sklearn.metrics import classification_report

def avaliar_modelo(model, X_test, y_test):
    resultado = model.evaluate(
        X_test,
        y_test,
        verbose=0,
        return_dict=True
    )

    print("\nAvaliação final do modelo:")

    for metrica, valor in resultado.items():
        print(f"{metrica}: {valor:.4f}")

    # Gerar as probabilidades previstas pelo modelo
    y_pred_prob = model.predict(X_test, verbose = 0)
    # Converter as probabilidades em classes 0 ou 1
    y_pred = (y_pred_prob >= 0.5).astype(int).ravel()
    # Exibir o relatório de classificação
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    return resultado, y_pred, y_pred_prob


# model.evaluate(): calcula a perda e as métricas configuradas na compilação.
# X_test e y_test: fornecem as características e os rótulos dos registros de teste.
# verbose=0: oculta a barra de progresso.
# return_dict=True: retorna as métricas em um dicionário.
# for metrica, valor in resultado.items(): percorre todas as métricas retornadas.
# model.predict(): gera as probabilidades previstas para os dados de teste.
# y_pred_prob: armazena as probabilidades previstas para a classe 1.
# >= 0.5: define o limite para classificar uma amostra como classe 1.
# astype(int): converte os resultados booleanos em inteiros 0 ou 1.
# ravel(): transforma as previsões em um array unidimensional.
# classification_report(): apresenta precision, recall, f1-score e support.
# return resultado: permite reutilizar as métricas em comparações e relatórios.