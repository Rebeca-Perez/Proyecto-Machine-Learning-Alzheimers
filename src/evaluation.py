import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, roc_auc_score

test_data = "../data/test/test.csv"

"""Ruta del modelo elegido, en este caso el Random Forest.
    Se puede cambiar la ruta para probar otros modelos"""

modelo = "../models/modelo_final/06_modelo_RandomForest.pkl"


"""Carga los datos de test."""

def cargar_test_data(path=test_data):
    df = pd.read_csv(path)
    X_test = df.drop(["Diagnosis"], axis=1)
    y_test = df["Diagnosis"]
    return X_test, y_test

"""Carga el modelo desde pickle."""

def cargar_modelo(path=modelo):
    modelo_final = pickle.load(open(path, "rb"))
    return modelo_final

"""Evalúa el modelo y genera métricas."""

def evaluate(modelo, X_test, y_test):
    pred = modelo.predict(X_test)

    # Intentar AUC (no todos los modelos tienen predict_proba)
    try:
        scores = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, scores)
    except:
        auc = "Modelo sin predict_proba"

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred)}
    return metrics

print("Finalizado")