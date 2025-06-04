
## Configuración y Despliegue

### Prerrequisitos
*   Cuenta de AWS
*   AWS CLI configurada localmente (para algunas operaciones manuales si son necesarias)
*   Docker instalado localmente (para construir/probar imágenes)
*   Git

### Configuración de AWS
Los siguientes recursos de AWS deben ser creados manualmente (o mediante IaC en una versión más avanzada):
*   Bucket S3: `mlops-winequality-9102` (Región: `us-east-2`)
    *   Con una carpeta `test_data/` conteniendo `test_features.csv` y `test_labels.csv`.
*   Repositorio ECR: `mlops-proj` (Región: `us-east-2`)
*   Rol IAM para Lambda (`lambda-wine-quality-api-dev-role` y uno similar para prod) con permisos para:
    *   `AWSLambdaBasicExecutionRole` (logs a CloudWatch)
    *   Leer `model.onnx` y `scaler.joblib` de S3.
    *   Escribir `predictions/wine_quality_dev.txt` (o `_prod.txt`) a S3.
*   Funciones Lambda (`wineQualityApi-dev` y `wineQualityApi-prod`) configuradas para imagen de contenedor, con el rol IAM y variables de entorno adecuadas.
*   API Gateways (`WineQualityDevAPI` y una similar para prod) con una etapa `dev` (o `prod`) apuntando a la Lambda correspondiente.

### Variables de Entorno y Secrets
**Para GitHub Actions (a configurar como Secrets del repositorio):**
*   `AWS_ACCESS_KEY_ID`: Access Key ID del usuario IAM con permisos para S3, ECR, Lambda.
*   `AWS_SECRET_ACCESS_KEY`: Secret Access Key correspondiente.
*   `AWS_REGION`: `us-east-2`
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`
*   `ECR_REPOSITORY_URI`: `913513915525.dkr.ecr.us-east-2.amazonaws.com/mlops-proj`
*   `LAMBDA_FUNCTION_NAME_DEV`: `wineQualityApi-dev`
*   `LAMBDA_FUNCTION_NAME_PROD`: `wineQualityApi-prod` (a ser creada)

**Variables de Entorno para las Funciones Lambda (configuradas en Lambda):**
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`
*   `S3_MODEL_KEY_ONNX`: `wine_quality_model.onnx`
*   `S3_SCALER_KEY`: `wine_quality_scaler.joblib`
*   `PREDICTION_LOG_KEY_PREFIX`: `predictions/wine_quality`
*   `ENVIRONMENT_NAME`: `dev` (para la Lambda de dev) o `prod` (para la Lambda de prod).

## Uso de la API

Una vez desplegada, la API se puede acceder a través de la URL de invocación proporcionada por API Gateway.

**Endpoint Raíz (Health Check):**
*   `GET {URL_BASE_API}/`
*   Respuesta: `{"message": "Welcome to the Wine Quality Prediction API (dev/prod). Model and scaler are loaded."}`

**Endpoint de Predicción:**
*   `POST {URL_BASE_API}/predict`
*   **Cuerpo de la solicitud (JSON):**
    ```json
    {
      "alcohol": 13.2,
      "malic_acid": 1.78,
      // ... todas las 13 características del vino
      "proline": 1050.0
    }
    ```
*   **Respuesta Exitosa (200 OK):**
    ```json
    {
      "predicted_quality_class": 0 
    }
    ```
    (La clase predicha puede ser 0, 1, o 2)

Cada predicción exitosa se registrará en `s3://mlops-winequality-9102/predictions/wine_quality_{ENVIRONMENT_NAME}.txt`.

## Desafíos y Consideraciones

### Cold Starts en AWS Lambda
Las funciones AWS Lambda que utilizan imágenes de contenedor, especialmente aquellas con dependencias de Machine Learning que resultan en imágenes de tamaño considerable (ej. ~500-600MB en este proyecto), pueden experimentar "cold starts" significativos. La fase de inicialización (`init`) de Lambda tiene un límite de tiempo estricto (actualmente 10 segundos). Si la descarga y descompresión de la imagen de ECR, junto con la inicialización del runtime y el código de la aplicación, excede este límite, la primera invocación puede fallar por `init timeout`.

**Mitigación para la Demostración:**
*   La Lambda se ha configurado con memoria aumentada (ej. 1024MB-2048MB) y un tiempo de espera de invocación más largo (ej. 1-2 minutos).
*   Para la demostración, puede ser necesario "calentar" la Lambda invocando el endpoint de API Gateway varias veces. Las primeras peticiones podrían encontrar un timeout de API Gateway (29s) si la `init` de Lambda es muy larga. Sin embargo, una vez que una instancia de Lambda se inicializa completamente (lo que se puede observar en los logs de CloudWatch), las invocaciones subsiguientes a esa instancia "caliente" serán rápidas y funcionales.

Este es un desafío conocido y una consideración importante para aplicaciones de ML en arquitecturas serverless con Lambda. Soluciones de producción podrían incluir Provisioned Concurrency (con costos asociados) o una optimización más agresiva del tamaño de la imagen.

## Futuras Mejoras
*   Implementar Provisioned Concurrency para la Lambda de `prod` para eliminar cold starts.
*   Optimizar aún más el tamaño de la imagen Docker (explorar alternativas más ligeras para dependencias si es posible).
*   Añadir pruebas de integración más completas.
*   Implementar un sistema de versionado de modelos más robusto.
*   Añadir monitoreo de modelos (deriva de datos, deriva de concepto).
*   Usar Infraestructura como Código (IaC) como AWS CDK o Terraform para definir los recursos de AWS.
*   Implementar un pipeline de reentrenamiento automático basado en triggers.

---