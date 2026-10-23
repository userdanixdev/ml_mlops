# Compilação, treino e avaliação do Modelo:
from tensorflow import keras
from sklearn.metrics import classification_report
from mlops_moon import (
    model, 
    X_train,
    y_train,
    X_test,
    y_test
)

# Compilação do modelo:
# model = keras.Model(inputs,outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)
class HistoricoEpocas(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        print(
            f"Época [{epoch + 1}/{self.params['epochs']}] - "
            f"Loss: [{logs.get('loss',0):.6f} - "
            f"Val_Loss: [{logs.get('val_loss',0):.6f} "
        )
# Treinamento:
hist = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=0,
    callbacks=[HistoricoEpocas()]
)

# Avaliação final com os dados de teste
resultado = model.evaluate(
    X_test,
    y_test,
    verbose=0,
    return_dict=True
)

#print(f"Loss de teste: {loss_test:.4f}")
#print(f"Acurácia de teste: {accuracy_test:.4f}")

# Relatório de classificação
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

print(classification_report(y_test, y_pred))