import os
import pandas as pd
import joblib
import onnxruntime
import numpy as np
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import accuracy_score
import tempfile

# --- Configuración de S3 y Nombres de Clave/Archivo ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
# Artefactos del modelo (los que subió train.py)
S3_MODEL_KEY_ONNX = os.environ.get("S3_MODEL_KEY_ONNX", "wine_quality_model.onnx")
S3_SCALER_KEY = os.environ.get("S3_SCALER_KEY", "wine_quality_scaler.joblib")
# Datos de prueba (los que subiste manualmente)
S3_TEST_FEATURES_KEY = os.environ.get("S3_TEST_FEATURES_KEY", "test_data/test_features.csv")
S3_TEST_LABELS_KEY = os.environ.get("S3_TEST_LABELS_KEY", "test_data/test_labels.csv")

# Umbral para la prueba de métrica
MINIMUM_ACCURACY_THRESHOLD = 0.70 # Ajusta según sea necesario

# Rutas temporales locales
TEMP_DIR = tempfile.gettempdir()
LOCAL_MODEL_PATH = os.path.join(TEMP_DIR, "test_model.onnx")
LOCAL_SCALER_PATH = os.path.join(TEMP_DIR, "test_scaler.joblib")
LOCAL_TEST_FEATURES_PATH = os.path.join(TEMP_DIR, "test_features.csv")
LOCAL_TEST_LABELS_PATH = os.path.join(TEMP_DIR, "test_labels.csv")

s3_client = None

def download_from_s3(bucket, key, local_path):
    global s3_client
    if not s3_client:
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

def run_tests():
    print("--- Starting Model Tests ---")

    if not S3_BUCKET_NAME:
        print("ERROR: S3_BUCKET_NAME environment variable not set.")
        return False # Falla la prueba

    # 1. Descargar artefactos y datos de prueba
    print("\n--- 1. Downloading Artefacts and Test Data ---")
    if not download_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY_ONNX, LOCAL_MODEL_PATH): return False
    if not download_from_s3(S3_BUCKET_NAME, S3_SCALER_KEY, LOCAL_SCALER_PATH): return False
    if not download_from_s3(S3_BUCKET_NAME, S3_TEST_FEATURES_KEY, LOCAL_TEST_FEATURES_PATH): return False
    if not download_from_s3(S3_BUCKET_NAME, S3_TEST_LABELS_KEY, LOCAL_TEST_LABELS_PATH): return False

    # 2. Cargar modelo, scaler y datos de prueba
    print("\n--- 2. Loading Model, Scaler, and Test Data ---")
    try:
        onnx_session = onnxruntime.InferenceSession(LOCAL_MODEL_PATH)
        scaler = joblib.load(LOCAL_SCALER_PATH)
        X_test_df = pd.read_csv(LOCAL_TEST_FEATURES_PATH)
        y_test_series = pd.read_csv(LOCAL_TEST_LABELS_PATH).squeeze("columns") # squeeze para convertir a Series
        print("Model, scaler, and test data loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load artefacts or test data: {e}")
        return False

    # 3. Prueba de Respuesta del Modelo (con la primera fila de datos de prueba)
    print("\n--- 3. Testing Model Responsiveness ---")
    try:
        sample_input_df = X_test_df.head(1).copy()
        
        if 'od280_od315_of_diluted_wines' in sample_input_df.columns:
            sample_input_df.rename(columns={'od280_od315_of_diluted_wines': 'od280/od315_of_diluted_wines'}, inplace=True)
        
        if hasattr(scaler, 'feature_names_in_'):
             ordered_sample_input_df = sample_input_df[scaler.feature_names_in_]
        else:
             ordered_sample_input_df = sample_input_df

        scaled_sample_features = scaler.transform(ordered_sample_input_df)
        
        input_name = onnx_session.get_inputs()[0].name
        input_feed = {input_name: scaled_sample_features.astype(np.float32)}
        result = onnx_session.run(None, input_feed)
        
        prediction = int(result[0][0])
        print(f"Model responded with prediction: {prediction} for the first test sample.")
        if not isinstance(prediction, int):
            print(f"ERROR: Model prediction is not an integer: {prediction}")
            return False
    except Exception as e:
        print(f"ERROR: Model responsiveness test failed: {e}")
        return False

    # 4. Prueba de Métrica (Accuracy)
    print("\n--- 4. Testing Model Metric (Accuracy) ---")
    try:
        X_test_processed_df = X_test_df.copy()
        if 'od280_od315_of_diluted_wines' in X_test_processed_df.columns:
            X_test_processed_df.rename(columns={'od280_od315_of_diluted_wines': 'od280/od315_of_diluted_wines'}, inplace=True)
        
        if hasattr(scaler, 'feature_names_in_'):
            ordered_X_test_df = X_test_processed_df[scaler.feature_names_in_]
        else:
            ordered_X_test_df = X_test_processed_df

        X_test_scaled = scaler.transform(ordered_X_test_df)
        
        all_predictions = []
        input_name = onnx_session.get_inputs()[0].name
        for i in range(len(X_test_scaled)):
            row_scaled = X_test_scaled[i:i+1]
            input_feed = {input_name: row_scaled.astype(np.float32)}
            result_row = onnx_session.run(None, input_feed)
            all_predictions.append(int(result_row[0][0]))
            
        accuracy = accuracy_score(y_test_series, all_predictions)
        print(f"Model accuracy on test data: {accuracy:.4f}")
        
        if accuracy >= MINIMUM_ACCURACY_THRESHOLD:
            print(f"PASSED: Accuracy ({accuracy:.4f}) is >= threshold ({MINIMUM_ACCURACY_THRESHOLD:.4f}).")
        else:
            print(f"FAILED: Accuracy ({accuracy:.4f}) is < threshold ({MINIMUM_ACCURACY_THRESHOLD:.4f}).")
            return False
            
    except Exception as e:
        print(f"ERROR: Model metric test failed: {e}")
        return False

    print("\n--- All Model Tests Passed Successfully! ---")
    return True


if __name__ == "__main__":
    tests_passed = run_tests()
    if tests_passed:
        print("Script finished: TESTS PASSED")
        exit(0)
    else:
        print("Script finished: TESTS FAILED")
        exit(1)