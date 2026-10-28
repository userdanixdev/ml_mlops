# Compilação e treinamento - Dataset Moon
def treinar_modelo(model, X_train, y_train):
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    # Quantidade de épocas executadas:
    total_epochs = len(history.epoch)
    
    # Identificar a época com a menor perda de validação:
    val_loss = history.history["val_loss"]
    melhor_indice = val_loss.index(min(val_loss))
    melhor_epoch = melhor_indice + 1
    menor_val_loss = val_loss[melhor_indice]
    
    # Recuperar as métricas da melhor época:
    loss_treino = history.history["loss"][melhor_indice]
    accuracy_treino = history.history["accuracy"][melhor_indice]
    accuracy_validacao = history.history["val_accuracy"][melhor_indice]
    
    # Exibição do treinamento:
    print("\n ======= Resumo do Treinamento ======")
    print(f"Total de épocas executadas: {total_epochs}")
    print(f"Melhor Época: {melhor_epoch}")
    print(f"Menor val_loss: {menor_val_loss:.4f}")
    print(f"Loss de treino na melhor época: {loss_treino:.4f}")
    print(f"Acurácia de treino da melhor época: {accuracy_treino:.4f}")
    print(f"Acurácia de validação: {accuracy_validacao:.4f}")

    # Verificação simples de possíveis sinais de overfitting
    loss_validacao = history.history["val_loss"]
    sinais_overfitting = (
        loss_treino < menor_val_loss
        and melhor_epoch < total_epochs
        and loss_validacao[-1] > menor_val_loss
    )
    # Exibir análise de overfitting
    largura = 60

    print("\n" + "=" * largura)
    print("ANÁLISE DE OVERFITTING".center(largura))
    print("=" * largura)

    if sinais_overfitting:
        print("Status: POSSÍVEIS SINAIS IDENTIFICADOS")
        print("-" * largura)
        print("A perda de validação aumentou após atingir")
        print("seu menor valor durante o treinamento.")
    else:
        print("Status: NENHUM SINAL CLARO IDENTIFICADO")
        print("-" * largura)
        print("A regra utilizada não identificou sinais")
        print("claros de overfitting.")

    print("=" * largura + "\n")    
        
    return history