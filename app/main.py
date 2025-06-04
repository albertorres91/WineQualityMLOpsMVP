import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import onnxruntime
import numpy as np
#import pandas as pd
import boto3 
from botocore.exceptions import ClientError # Para manejo de errores de S3
from datetime import datetime # Para el log de predicciones
import tempfile
import time

print(f"[{time.time():.2f}] app/main.py: MODULE LEVEL EXECUTION STARTED") # NUEVO PRINT

# --- Configuración de S3 y Nombres de Archivo/Clave ---
# Estas se pasarán como variables de entorno a Lambda/Contenedor
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_MODEL_KEY_ONNX = os.environ.get("S3_MODEL_KEY_ONNX", "wine_quality_model.onnx") # Default si no se establece
S3_SCALER_KEY = os.environ.get("S3_SCALER_KEY", "wine_quality_scaler.joblib") # Default si no se establece
PREDICTION_LOG_KEY_PREFIX = os.environ.get("PREDICTION_LOG_KEY_PREFIX", "predictions/wine_quality") # ej. predictions/wine_quality_dev.txt
ENVIRONMENT_NAME = os.environ.get("ENVIRONMENT_NAME", "local") # 'dev' o 'prod' en la nube

# --- Rutas temporales locales ---
# Usaremos tempfile para crear nombres de archivo únicos en el directorio temporal del sistema
TEMP_DIR = tempfile.gettempdir() # Obtiene el directorio temporal del sistema (ej. C:\Users\...\AppData\Local\Temp en Windows)
LOCAL_MODEL_PATH = os.path.join(TEMP_DIR, "downloaded_model.onnx")
LOCAL_SCALER_PATH = os.path.join(TEMP_DIR, "downloaded_scaler.joblib")

'''
# --- Configuración de Rutas (Forma Robusta) ---
# Obtener el directorio del script actual (WineQualityMLOpsMVP/app/)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta a la raíz del proyecto subiendo un nivel desde APP_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))

# --- Configuración ---
# Para desarrollo local, cargaremos desde el sistema de archivos.
# Asumimos que train.py ha generado estos archivos en el directorio raíz del proyecto.
# Si train.py los guarda en otro sitio (ej. ./scripts/), ajusta las rutas.
MODEL_FILENAME = "model.onnx"
SCALER_FILENAME = "scaler.joblib"
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILENAME)
SCALER_PATH = os.path.join(PROJECT_ROOT, SCALER_FILENAME)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] APP_DIR: {APP_DIR}")

# Construir la ruta subiendo un nivel desde APP_DIR
PROJECT_ROOT_GUESS = os.path.abspath(os.path.join(APP_DIR, os.pardir))
print(f"[DEBUG] PROJECT_ROOT_GUESS: {PROJECT_ROOT_GUESS}")

MODEL_PATH = os.path.join(PROJECT_ROOT_GUESS, "model.onnx")
SCALER_PATH = os.path.join(PROJECT_ROOT_GUESS, "scaler.joblib")

print(f"[DEBUG] Absolute MODEL_PATH attempting to load: {MODEL_PATH}")
print(f"[DEBUG] Absolute SCALER_PATH attempting to load: {SCALER_PATH}")
'''

# Variables globales para el modelo y el scaler
onnx_session = None
scaler = None
s3_client = None # Para S3


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

def download_from_s3(bucket, key, local_path):
    global s3_client
    if not s3_client: # Inicializar cliente S3 si no existe
        s3_client = boto3.client("s3")
    try:
        print(f"Attempting to download s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        print(f"Successfully downloaded {key} to {local_path}")
        return True
    except ClientError as e:
        print(f"Error downloading {key} from S3: {e}")
        if e.response['Error']['Code'] == '404':
            print(f"The object s3://{bucket}/{key} does not exist.")
        elif e.response['Error']['Code'] == '403':
            print(f"Access denied for s3://{bucket}/{key}. Check IAM permissions.")
        else:
            print(f"An unexpected error occurred: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"Generic error downloading {key} from S3: {e}")
        return False
    
print(f"[{time.time():.2f}] app/main.py: About to define startup_event") # NUEVO PRINT

@app.on_event("startup")
def startup_event(): # Renombrado para claridad y para añadir más lógica de startup
    global onnx_session, scaler, s3_client
    print(f"[{time.time():.2f}] Startup event started.")
    
    
    print(f"[INFO] Local model path will be: {LOCAL_MODEL_PATH}") # Log para ver la ruta temporal
    print(f"[INFO] Local scaler path will be: {LOCAL_SCALER_PATH}") # Log para ver la ruta temporal

    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME environment variable not set.")
    
    s3_client = boto3.client("s3") # Inicializar cliente S3

    # Descargar modelo ONNX
    start_dl_model = time.time()
    if not download_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY_ONNX, LOCAL_MODEL_PATH):
        raise RuntimeError(f"Failed to download ONNX model from S3. Check logs.")
    print(f"[{time.time():.2f}] Model downloaded in {time.time() - start_dl_model:.2f}s.")
    
    # Descargar scaler
    start_dl_scaler = time.time()
    if not download_from_s3(S3_BUCKET_NAME, S3_SCALER_KEY, LOCAL_SCALER_PATH):
        raise RuntimeError(f"Failed to download scaler from S3. Check logs.")
    print(f"[{time.time():.2f}] Scaler downloaded in {time.time() - start_dl_scaler:.2f}s.")

    start_load_model = time.time()
    try:
        print(f"Loading ONNX model from {LOCAL_MODEL_PATH}")
        onnx_session = onnxruntime.InferenceSession(LOCAL_MODEL_PATH)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model from {LOCAL_MODEL_PATH}: {e}")
        raise RuntimeError(f"Error loading ONNX model: {e}")
    print(f"[{time.time():.2f}] ONNX model loaded in {time.time() - start_load_model:.2f}s.")

    start_load_scaler = time.time()
    try:
        print(f"Loading scaler from {LOCAL_SCALER_PATH}")
        scaler = joblib.load(LOCAL_SCALER_PATH)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler from {LOCAL_SCALER_PATH}: {e}")
        raise RuntimeError(f"Error loading scaler: {e}")
    print(f"[{time.time():.2f}] Scaler loaded in {time.time() - start_load_scaler:.2f}s.")

    if onnx_session is None:
        raise RuntimeError("ONNX session could not be initialized.")
    if scaler is None:
        raise RuntimeError("Scaler could not be loaded.")
    
    print(f"[{time.time():.2f}] Application startup complete.")


@app.get("/", tags=["Health Check"])
async def root():
    if onnx_session and scaler:
        return {"message": f"Welcome to the Wine Quality Prediction API ({ENVIRONMENT_NAME}). Model and scaler are loaded."}
    else:
        return {"message": f"Welcome to the Wine Quality Prediction API ({ENVIRONMENT_NAME}). Error: Model or scaler not loaded."}
    
    
# --- Implementación del Logging de Predicciones ---
def log_prediction_to_s3(prediction_data: str):
    global s3_client
    if not s3_client:
        s3_client = boto3.client("s3")

    log_file_key = f"{PREDICTION_LOG_KEY_PREFIX}_{ENVIRONMENT_NAME}.txt"
    
    try:
        # Intentar obtener el contenido actual del archivo de log
        try:
            existing_log_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=log_file_key)
            existing_log_content = existing_log_object['Body'].read().decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                existing_log_content = "" # El archivo no existe aún, empezar vacío
                print(f"Log file s3://{S3_BUCKET_NAME}/{log_file_key} not found. Creating a new one.")
            else:
                raise # Otro error de S3
        
        # Añadir la nueva predicción y una marca de tiempo
        timestamp = datetime.utcnow().isoformat()
        new_log_entry = f"{timestamp}Z - {prediction_data}\n"
        updated_log_content = existing_log_content + new_log_entry
        
        # Subir el contenido actualizado
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=log_file_key, Body=updated_log_content.encode('utf-8'), ContentType='text/plain')
        print(f"Prediction logged to s3://{S3_BUCKET_NAME}/{log_file_key}")

    except Exception as e:
        print(f"Error logging prediction to S3 (s3://{S3_BUCKET_NAME}/{log_file_key}): {e}")
        # No relanzar la excepción aquí para no interrumpir la respuesta al usuario si el log falla


@app.post("/predict", tags=["Prediction"])
async def predict_quality(features: WineFeatures):
    global onnx_session, scaler

    if onnx_session is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded. Check server logs.")

    try:
        # 1. Convertir Pydantic model a un diccionario
        input_data_dict = features.model_dump()
        # print(f"[DEBUG PREDICT] Input data dict: {input_data_dict}") # Para depuración si es necesario

        # 2. Definir el orden esperado de características (DEBE COINCIDIR CON EL ENTRENAMIENTO DEL SCALER)
        # Este es el orden del dataset load_wine() de scikit-learn
        feature_order = [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280/od315_of_diluted_wines', # Nombre con barra, como lo espera el scaler
            'proline'
        ]
        
        # 3. Construir la lista de valores en el orden correcto
        input_values_ordered = []
        for feature_name_in_order in feature_order:
            # El modelo Pydantic usa '_' donde el nombre original de la característica tiene '/'
            pydantic_key = feature_name_in_order.replace('/', '_')
            
            if pydantic_key in input_data_dict:
                input_values_ordered.append(input_data_dict[pydantic_key])
            else:
                # Esto indica un problema con la definición de WineFeatures o feature_order
                print(f"[ERROR PREDICT] Missing feature {pydantic_key} (expected from {feature_name_in_order}) in input_data_dict.")
                raise ValueError(f"Missing critical feature for model: {pydantic_key}")

        # print(f"[DEBUG PREDICT] Input values ordered: {input_values_ordered}") # Para depuración

        # 4. Convertir a un array NumPy 2D (1 fila, N columnas) y asegurar tipo float32
        input_array = np.array([input_values_ordered], dtype=np.float32)
        # print(f"[DEBUG PREDICT] Input NumPy array shape: {input_array.shape}, dtype: {input_array.dtype}") # Para depuración

        # 5. Escalar las características con el array NumPy
        # scaler.transform() espera un array 2D
        scaled_features_np = scaler.transform(input_array)
        # print(f"[DEBUG PREDICT] Scaled features NumPy array shape: {scaled_features_np.shape}, dtype: {scaled_features_np.dtype}") # Para depuración
        
        # 6. Preparar entrada para el modelo ONNX
        input_name = onnx_session.get_inputs()[0].name
        input_feed = {input_name: scaled_features_np} # scaled_features_np ya es float32 por el input_array
        
        # 7. Realizar la predicción
        result = onnx_session.run(None, input_feed)
        prediction_class = int(result[0][0]) # La clase predicha

        # 8. Loguear la predicción
        log_data = f"Input: {input_data_dict}, Prediction: {prediction_class}"
        log_prediction_to_s3(log_data)

        return {"predicted_quality_class": prediction_class}

    except ValueError as ve: # Capturar errores de valor específicos, como claves faltantes
        print(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=422, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        # Loguear el error también podría ser una buena idea si no es un ValueError ya manejado
        print(f"Generic error during prediction: {e}")
        # Considera si quieres loguear este error a S3 también, o solo a CloudWatch.
        # log_prediction_to_s3(f"ERROR during prediction: Input: {features.model_dump()}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")