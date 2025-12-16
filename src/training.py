import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

process_data = "../data/processed/datos_limpios.csv"
train_data = "../data/train/train.csv"
test_data = "../data/test/test.csv"
modelo = "../models/06_modelo_RandomForest.pkl"

"""Carga de los datos limpios."""

def cargar_process_data(path=process_data):
    return pd.read_csv(path)

"""Define X, y y separa en train y test."""

def separar_datos(df):
    X = df.drop(["Diagnosis"], axis=1)
    y = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
    return X_train, X_test, y_train, y_test

"""Guarda los CSV de train y test."""

def guardar_dataset(X_train, X_test, y_train, y_test):
    os.makedirs(os.path.dirname(train_data), exist_ok=True)
    os.makedirs(os.path.dirname(test_data), exist_ok=True)

    train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)

    train.to_csv(train_data, index=False)
    test.to_csv(test_data, index=False)


"""Entrena el modelo Random Forest."""

def entrenar_modelo(X_train, y_train):
    rf = RandomForestClassifier(random_state=11)

    parametros = {
        "n_estimators": [400, 700, 800],
        "max_depth": [4, 5, 6]}

    gs_rf = GridSearchCV(
        estimator=rf,
        param_grid=parametros,
        scoring="recall",
        cv=3,
        verbose=3,
        n_jobs=-1)

    gs_rf.fit(X_train, y_train)
    modelo_final = gs_rf.best_estimator_
    
    return modelo_final

"""Guarda el modelo con pickle."""

def guardar_modelo(modelo, path=modelo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(modelo, open(path, "wb"))
