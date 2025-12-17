# 🧠 Proyecto de Machine Learning — Diagnóstico de Alzheimer

Este proyecto se basa en utilizar Machine Learning para predecir la variable **Diagnosis** (diagnóstico de Alzheimer) a partir de datos clínicos.  
Se ha diseñado siguiendo una estructura modular mediante scripts en la carpeta `src/`, permitiendo:

- Procesar datos
- Entrenar modelos
- Evaluar resultados

## 📁 Estructura del Proyecto

Proyecto-Machine-Learning-Alzheimers/  

app_streamlit/  
    img/ # imagen utilizada en streamlit  
    app.py #  
    requirements.txt #  

data/  
    raw/ # Datos originales  
    processed/ # Datos limpios generados por data_processing.py  
    train/ # Datos de entrenamiento generados por training.py  
    test/ # Datos de test generados por training.py  

docs/  
    img/ # imagen utilizada para la memoria  
    Memoria.ipynb # Archivo con la memoria del proyecto  
    Presentacion_técnica.pdf # Presentación del proyecto para Ciencia de Datos  
    Presentacion_negocio.pdf # Presentación del proyecto enfocado a negocio  
    
models/  
    modelos_probados/ # Modelos entrenados guardados en pickle  
    modelo_final/ # Modelo elegido, entrenado y guardado en pickle  
    model_config.yaml # librerías utilizadas y parámetros del modelo elegido  

notebooks/  
    01_Fuentes # Archivo jupyter con los datos iniciales  
    02_Limpieza # Archivo jupyter con limpieza de datos  
    03_Separar_datos_train_test # Archivo jupyter con separación de datos  
    04_Entrenamiento_LogisticRegression # Archivo jupyter con separado en train y test, guardado en archivos csv y entrenamiento y pruebas de modelo Logistic Regression  
    05_Entrenamiento_DecisionTree # Archivo jupyter con entrenamiento y pruebas de modelo Decision Tree utilizando GridSearch  
    06_Entrenamiento_SVC # Archivo jupyter con entrenamiento y pruebas de modelo SVC utilizando GridSearch  
    07_Entrenamiento_AdaBoost # Archivo jupyter con entrenamiento y pruebas de modelo Ada Boost utilizando GridSearch  
    08_Entrenamiento_GradientBoost # Archivo jupyter con entrenamiento y pruebas de modelo Gradient Boost utilizando GridSearch  
    09_Entrenamiento_RandomForest # Archivo jupyter con entrenamiento y pruebas de modelo Random Forest utilizando GridSearch  
    10_Entrenamiento_Kmeans # Archivo jupyter con entrenamiento y pruebas de modelo Kmeans como preprocesamiento y probado con Random Forest y Logistic Regression  
    11_Entrenamiento_RedNeuronal # Archivo jupyter con entrenamiento y pruebas de modelo Sequencial de redes neuronales  
    

src/  
    data_processing.py # Procesado y limpieza de datos  
    training.py # División train/test y entrenamiento de modelos  
    evaluation.py # Evaluación del modelo final  
    main.py # ejecución de los .py anteriores  

README.md # Este archivo

## 1. `data_processing.py` — Procesado y Limpieza de Datos

Este script:

- Carga los datos originales desde `data/raw/`
- Aplica la limpieza utilizada en el análisis
- Elimina las columnas irrelevantes o redundantes
- Genera el archivo limpio

## 2. `training.py` — Entrenamiento del Modelo

Este script:

- Carga de datos limpios
- División en train y test con train_test_split
- Generación de:  
      - data/train/train.csv  
      - data/test/test.csv  
- Entrenamiento del modelo final elegido (Ada Boost)
- Guardado del modelo entrenado en:  
      - models/modelo_final.pkl

## 3. `evaluation.py` — Evaluación del Modelo

Este script:

- Carga test.csv
- Carga el modelo indicado
- Calcula métricas:  
      - Accuracy  
      - Recall  
      - Matriz de confusión  
      - Classification Report  
      - AUC (si el modelo tiene predict_proba)  

Para evaluar otro modelo basta con cambiar esta línea:  
modelo = "../models/otro_modelo.pkl"

📌 Modelos Adicionales

Aunque el modelo principal se ejecuta desde main.py,
el proyecto incluye notebooks adicionales donde se entrenan otros modelos como:

- Logistic Regression
- DecisionTree
- Gradient Boosting
- Kmeans
- RandomForest
- Redes Neuronales
- SVM

Cada uno se guarda también en la carpeta models/

🛠 Tecnologías Utilizadas

Python 3.x

pandas

scikit-learn

pickle

keras

seaborn / matplotlib (visualizaciones auxiliares)

## Autor

Rebeca Pérez
Proyecto de Machine Learning — Diagnóstico de Alzheimer
