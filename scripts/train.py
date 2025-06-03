import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # O RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import boto3
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_wine

# --- Configuración ---
# Estas variables se obtendrán de las variables de entorno en GitHub Actions
# Para pruebas locales, puedes definirlas aquí o establecerlas en tu entorno
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_MODEL_KEY_ONNX = os.environ.get("S3_MODEL_KEY_ONNX", "wine_quality_model.onnx")
S3_SCALER_KEY = os.environ.get("S3_SCALER_KEY", "wine_quality_scaler.joblib")
MODEL_FILENAME_ONNX = "model.onnx"
SCALER_FILENAME = "scaler.joblib"

def load_and_preprocess_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    # Para simplificar, convertimos a un problema de clasificación binaria si hay más de 2 clases
    # O podrías filtrar por un tipo de vino si 'wine type' es una columna (dataset original de calidad de vino tiene esto)
    # Aquí, load_wine tiene 3 clases (0, 1, 2). Vamos a hacer un ejemplo de clasificación multiclase.
    # Si quisieras binario (ej. calidad > 5 vs <=5), tendrías que ajustar 'target'
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_model(X_train, y_train):
    # model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42) # Para multiclase
    model = LogisticRegression(max_iter=2000, random_state=42) # Simplificado, asumiendo que se ajustará
    # Si usas un dataset binario o quieres probar RandomForest:
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.4f}")
    return acc

def save_scaler_local(scaler, filename):
    print(f"Saving scaler to {filename}")
    joblib.dump(scaler, filename)

def convert_and_save_model_onnx(model, X_sample_shape_cols, filename):
    print(f"Converting model to ONNX and saving to {filename}")
    # El tipo inicial debe coincidir con la forma de una muestra de entrada
    # Para el dataset wine, hay 13 características
    initial_type = [('float_input', FloatTensorType([None, X_sample_shape_cols]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12) # Ajusta target_opset si es necesario
    
    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Model saved as ONNX.")

def upload_to_s3(bucket_name, local_filename, s3_key):
    if not bucket_name:
        print("S3_BUCKET_NAME no está configurado. Saltando subida a S3.")
        return
    
    s3_client = boto3.client("s3")
    try:
        print(f"Uploading {local_filename} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(local_filename, bucket_name, s3_key)
        print("Upload successful.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e

if __name__ == "__main__":
    print("Starting training process...")
    
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
    print(f"Data loaded. Training features: {len(feature_names)}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    save_scaler_local(scaler, SCALER_FILENAME)
    # Usamos el número de características para definir la forma de entrada del modelo ONNX
    convert_and_save_model_onnx(model, len(feature_names), MODEL_FILENAME_ONNX)

    # Subir artefactos a S3
    upload_to_s3(S3_BUCKET_NAME, MODEL_FILENAME_ONNX, S3_MODEL_KEY_ONNX)
    upload_to_s3(S3_BUCKET_NAME, SCALER_FILENAME, S3_SCALER_KEY)
    
    print("Training process finished.")