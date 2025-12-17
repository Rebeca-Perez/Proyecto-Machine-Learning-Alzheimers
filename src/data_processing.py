import pandas as pd
import os

raw_data = "../data/raw/alzheimers_disease_data.csv"
process_data = "../data/processed/datos_limpios.csv"

"""Carga del dataset original."""

def cargar_raw_data(path=raw_data):
    return pd.read_csv(path)

"""Limpieza de datos antes de entrenar al modelo."""

def datos_procesados(df):
    df = df.drop([
        "DoctorInCharge", "PatientID", "CholesterolTotal", "Age", "Smoking",
        "AlcoholConsumption", "PhysicalActivity", "DietQuality", "Depression",
        "DiastolicBP", "DifficultyCompletingTasks", "Forgetfulness", 
        'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'SleepQuality',
        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
        'HeadInjury', 'Hypertension', 'SystolicBP', 'CholesterolLDL',
        'CholesterolHDL', 'CholesterolTriglycerides', 'Confusion',
        'Disorientation', 'PersonalityChanges'], axis=1)
    return df
    
"""Guarda el dataset procesado."""

def guardar_processed_data(df, path=process_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
print("Guardado")