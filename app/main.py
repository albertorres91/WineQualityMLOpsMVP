import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import onnxruntime
import numpy as np
import pandas as pd

# --- Configuración ---
# Para desarrollo local, cargaremos desde el sistema de archivos.
# Asumimos que train.py ha generado estos archivos en el directorio raíz del proyecto.
# Si train.py los guarda en otro sitio (ej. ./scripts/), ajusta las rutas.
MODEL_PATH = "../model.onnx" # Sube un nivel desde 'app' a la raíz del proyecto
SCALER_PATH = "../scaler.joblib" # Sube un nivel desde 'app' a la raíz del proyecto

# Variables globales para el modelo y el scaler
onnx_session = None
scaler = None

app = FastAPI(title="Wine Quality Prediction API")

# --- Modelo Pydantic para los datos de entrada ---
# Basado en las características del dataset load_wine()
# Estas deben coincidir con las columnas que usó tu modelo para entrenar,
# EXCEPTO la variable objetivo ('target').
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    # Ejemplo de cómo se vería un request JSON:
    # {
    #   "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
    #   "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
    #   "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
    #   "hue": 1.05, "od280_od315_of_diluted_wines": 3.40, "proline": 1050.0
    # }

@app.on_event("startup")
def load_model_and_scaler():
    global onnx_session, scaler
    
    # Comprobar si los archivos existen antes de cargarlos
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found at {SCALER_PATH}. Run train.py first.")

    try:
        print(f"Loading ONNX model from {MODEL_PATH}")
        onnx_session = onnxruntime.InferenceSession(MODEL_PATH)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        raise RuntimeError(f"Error loading ONNX model: {e}")

    try:
        print(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        raise RuntimeError(f"Error loading scaler: {e}")

    if onnx_session is None:
        raise RuntimeError("ONNX session could not be initialized.")
    if scaler is None:
        raise RuntimeError("Scaler could not be loaded.")


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Welcome to the Wine Quality Prediction API. Model and scaler are loaded."}

@app.post("/predict", tags=["Prediction"])
async def predict_quality(features: WineFeatures):
    global onnx_session, scaler

    if onnx_session is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded. Check server logs.")

    try:
        # Convertir Pydantic model a un DataFrame de Pandas, luego a un array NumPy
        # El orden de las columnas debe ser el mismo que se usó para entrenar el scaler y el modelo
        input_df = pd.DataFrame([features.model_dump()]) 
        

        # Escalar las características
        scaled_features = scaler.transform(input_df)
        
        # Preparar la entrada para el modelo ONNX
        # El nombre 'float_input' debe coincidir con el definido en train.py durante la conversión a ONNX
        input_name = onnx_session.get_inputs()[0].name
        input_feed = {input_name: scaled_features.astype(np.float32)}
        
        # Realizar la predicción
        result = onnx_session.run(None, input_feed)
        
        # El resultado de un modelo de clasificación ONNX suele ser [array_de_predicciones, array_de_probabilidades (opcional)]
        # Para LogisticRegression de scikit-learn, la primera salida son las etiquetas predichas.
        prediction = int(result[0][0]) # Tomamos la primera predicción del primer (y único) batch

        # El dataset load_wine() tiene clases 0, 1, 2.
        # Podrías mapear esto a etiquetas más descriptivas si quieres.
        # class_labels = {0: "Class_0", 1: "Class_1", 2: "Class_2"}
        # predicted_label = class_labels.get(prediction, "Unknown_Class")

        return {"predicted_quality_class": prediction}

    except Exception as e:
        # Loguear el error e
        print(f"Error during prediction: {e}")
        # Devolver un error HTTP genérico
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Para ejecutar localmente con uvicorn ---
# Si ejecutas este archivo directamente (python app/main.py), esto no funcionará bien sin uvicorn.
# Es mejor ejecutar desde la terminal en la raíz del proyecto:
# cd WineQualityMLOpsMVP
# python scripts/train.py  (para generar model.onnx y scaler.joblib si no existen)
# uvicorn app.main:app --reload --port 8000