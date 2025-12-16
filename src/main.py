import data_processing as dp
import training as t
import evaluation as ev

df = dp.cargar_raw_data(dp.raw_data)
procesar = dp.datos_procesados(df)
guardar = dp.guardar_processed_data(procesar)
df = t.cargar_process_data(t.process_data)

X_train, X_test, y_train, y_test = t.separar_datos(df)

t.guardar_dataset(X_train, X_test, y_train, y_test)

entrenar = t.entrenar_modelo(X_train, y_train)
t.guardar_modelo(entrenar, t.modelo)

X_test, y_test = ev.cargar_test_data(ev.test_data)

modelo_final = ev.cargar_modelo(ev.modelo)

metricas = ev.evaluate(modelo_final, X_test, y_test)
print(metricas)