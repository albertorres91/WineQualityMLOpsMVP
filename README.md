
## Configuración y Despliegue

### Prerrequisitos
*   Cuenta de AWS.
*   AWS CLI configurada localmente.
*   Docker instalado localmente.
*   Git.

### Configuración de AWS Manual
La siguiente infraestructura de AWS se configuró manualmente en la región `us-east-2`:
*   Bucket S3: `mlops-winequality-9102` (con `wine_quality_model.onnx`, `wine_quality_scaler.joblib`, y carpeta `test_data/` con sus archivos).
*   Repositorio ECR: `mlops-proj`.
*   Roles IAM: `mlops-user`, `lambda-wine-quality-api-dev-role`, `lambda-wine-quality-api-prod-role`.
*   Políticas IAM personalizadas: `S3AccessForWineQualityMLOps`, `LambdaWineQualityS3Access-Dev`, `LambdaWineQualityS3Access-Prod`, `GitHubActionsLambdaUpdateAccess`.
*   Funciones Lambda: `wineQualityApi-dev`, `wineQualityApi-prod` (configuradas para imagen de ECR, con roles, variables de entorno, memoria de 2048MB y timeout de 1 min 30 seg).
*   API Gateways: `WineQualityDevAPI` (etapa `dev`), `WineQualityProdAPI` (etapa `prod`).
*   Reglas de Protección de Rama en GitHub para `prod` (requiriendo PR y que el job `tests_suite` pase).

### Variables de Entorno y Secrets de GitHub
**Secrets del Repositorio GitHub (Settings > Secrets and variables > Actions):**
*   `AWS_ACCESS_KEY_ID`: `AKIA5JMNMESCT2ULRP45`
*   `AWS_SECRET_ACCESS_KEY`: (La secret key correspondiente)
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`

**Variables de Entorno para las Funciones Lambda (configuradas en la consola de Lambda):**
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`
*   `S3_MODEL_KEY_ONNX`: `wine_quality_model.onnx`
*   `S3_SCALER_KEY`: `wine_quality_scaler.joblib`
*   `PREDICTION_LOG_KEY_PREFIX`: `predictions/wine_quality`
*   `ENVIRONMENT_NAME`: `dev` o `prod` (según la función).

## Uso de la API

Endpoints (reemplazar `{URL_BASE_API}` con la URL de invocación de API Gateway correspondiente al entorno):

**Endpoint Raíz (Health Check):**
*   `GET {URL_BASE_API}/`
*   Respuesta Esperada: `{"message": "Welcome to the Wine Quality Prediction API (dev/prod). SIMPLIFIED STARTUP - Models NOT loaded."}` (o el mensaje original si se revierte la simplificación del `startup_event`).

**Endpoint de Predicción:**
*   `POST {URL_BASE_API}/predict`
*   Cuerpo (JSON): Ver ejemplo en `app/main.py` o pruebas anteriores.
*   Respuesta Esperada (200 OK): `{"predicted_quality_class": X}`

Cada predicción (si la API se inicializa completamente y procesa la solicitud) se registra en `s3://mlops-winequality-9102/predictions/wine_quality_{ENVIRONMENT_NAME}.txt`.

## Pruebas Implementadas

El pipeline de CI/CD incluye dos tipos de pruebas:

1.  **Pruebas Unitarias (`tests/test_example_utils.py`):**
    *   Ejecutadas por `pytest`.
    *   Prueban funciones Python simples y aisladas para verificar la lógica básica del código. Estas pruebas no interactúan con AWS, modelos de ML, o la API FastAPI directamente.
2.  **Pruebas de Integración del Modelo (`scripts/model_tester.py`):**
    *   Descarga el modelo ONNX, el preprocesador (scaler), y datos de prueba desde S3.
    *   **Prueba de Respuesta:** Verifica que el modelo puede procesar una entrada de ejemplo y producir una predicción con el formato esperado.
    *   **Prueba de Métrica:** Calcula la accuracy del modelo en el conjunto de datos de prueba y la compara contra un umbral predefinido. Si la accuracy es menor al umbral, la prueba falla.

Ambos conjuntos de pruebas deben pasar para que el pipeline proceda al despliegue.

## Desafíos y Consideraciones Importantes

### Cold Starts, Timeouts en AWS Lambda, y Límites de Concurrencia
El desafío más significativo encontrado en este proyecto es el manejo de los "cold starts" para las funciones AWS Lambda que utilizan imágenes de contenedor. La imagen Docker para esta aplicación, incluso después de optimizaciones (como la eliminación de `pandas`), tiene un tamaño aproximado de 556MB debido a las dependencias de Machine Learning (`scikit-learn`, `onnxruntime`, `numpy`).

**1. `init timeout` de Lambda:**
AWS Lambda tiene un límite estricto de **10 segundos para la fase de inicialización (`init`)** de una nueva instancia "fría". Esta fase incluye la descarga de la imagen desde ECR, su descompresión y la preparación del entorno de ejecución de Python. Para imágenes grandes como la nuestra, este proceso puede exceder consistentemente los 10 segundos.
*   **Observación:** Los logs de CloudWatch muestran repetidos `INIT_REPORT Init Duration: ~10000 ms Phase: init Status: timeout`. Esto ocurre antes de que el código de la aplicación (`app/main.py`) comience su ejecución significativa.

**2. Timeouts de Invocación en Instancias "Semi-Calientes":**
En algunos reintentos de Lambda o con instancias que logran pasar la fase `init`, la aplicación FastAPI *sí* logra iniciarse (indicado por logs "Application startup complete" y "Uvicorn running"). Sin embargo, la primera invocación real a estas instancias (que ejecuta el `startup_event` con descargas S3 y carga de modelos, más el procesamiento de la petición) también puede exceder el tiempo de espera configurado para la Lambda (ej. 1m 30s) o el límite de 29 segundos de API Gateway, resultando en un timeout para el cliente.

**3. Imposibilidad de Usar Provisioned Concurrency:**
Se exploró la Concurrencia Aprovisionada como una solución para eliminar los cold starts. Sin embargo, debido a los límites de concurrencia predeterminados en la cuenta de AWS utilizada, no fue posible aprovisionar instancias dedicadas sin solicitar un aumento de cuota de servicio.

**Estrategia de Mitigación para la Demostración y Uso:**
*   **"Calentar" la Lambda:** Antes de una demostración o para pruebas consistentes, es necesario invocar el endpoint de API Gateway varias veces. Las primeras peticiones pueden fallar o dar timeout en el cliente. Eventualmente, una instancia de Lambda se inicializará completamente.
*   **Configuración de Lambda:** Las funciones Lambda se han configurado con memoria aumentada (2048 MB) y un tiempo de espera de invocación extendido (1 min 30 seg) para dar el máximo margen posible una vez que la fase `init` logra pasar.
*   **Invocaciones Posteriores:** Una vez que una instancia está "caliente" y la aplicación FastAPI se ha inicializado completamente, las invocaciones subsiguientes a esa misma instancia deberían ser rápidas y funcionales.
*   **Documentación:** Este comportamiento es una limitación conocida y se documenta aquí.

Este desafío subraya la importancia crítica de la optimización del tamaño de la imagen y la cuidadosa gestión de los cold starts en arquitecturas serverless para aplicaciones de Machine Learning.

## Futuras Mejoras
*   **Optimización Agresiva del Tamaño de la Imagen Docker.**
*   **AWS Lambda Provisioned Concurrency** (requiere aumento de límites de cuenta o presupuesto).
*   **Infraestructura como Código (IaC).**
*   **Monitoreo Avanzado.**
*   **Seguridad Mejorada.**
*   **Pipeline de Reentrenamiento.**
*   **Explorar Despliegue en EC2/ECS como alternativa para el rendimiento de inicio.**

---