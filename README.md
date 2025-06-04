# Wine Quality MLOps MVP Project

Este proyecto es un Producto Mínimo Viable (MVP) para un sistema de MLOps diseñado para el despliegue automático de modelos de clasificación de calidad de vino. Ha sido desarrollado como parte de la materia de MLOps.

**Estado Actual:** El pipeline de CI/CD está completamente funcional, automatizando el entrenamiento (simulado), pruebas del modelo, construcción de la imagen Docker, subida a ECR y despliegue a entornos `dev` y `prod` en AWS Lambda con API Gateway. Se ha identificado un desafío significativo con los "cold starts" de AWS Lambda debido al tamaño de la imagen del contenedor, lo que puede causar timeouts en la fase de inicialización y en la primera invocación.

## Tabla de Contenidos
1. [Visión General del Proyecto](#visión-general-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Flujo de Trabajo de CI/CD](#flujo-de-trabajo-de-cicd)
4. [Estructura del Repositorio](#estructura-del-repositorio)
5. [Configuración y Despliegue](#configuración-y-despliegue)
    - [Prerrequisitos](#prerrequisitos)
    - [Configuración de AWS Manual](#configuración-de-aws-manual)
    - [Variables de Entorno y Secrets de GitHub](#variables-de-entorno-y-secrets-de-github)
6. [Uso de la API](#uso-de-la-api)
7. [Desafíos y Consideraciones Importantes](#desafíos-y-consideraciones-importantes)
    - [Cold Starts y Timeouts en AWS Lambda](#cold-starts-y-timeouts-en-aws-lambda)
8. [Futuras Mejoras](#futuras-mejoras)

## Visión General del Proyecto

El objetivo principal de este proyecto es demostrar un pipeline de CI/CD completo para el despliegue automático de un modelo de Machine Learning. Cuando se actualiza el código o se "entrena" un nuevo modelo (simulado por el script de entrenamiento), el sistema automáticamente:

1.  Entrena un modelo de clasificación para la calidad del vino (usando el dataset "wine quality" de Scikit-learn).
2.  Guarda el modelo preprocesador (scaler) y el modelo de predicción en formato ONNX en un bucket de Amazon S3.
3.  Ejecuta pruebas sobre el modelo recién almacenado para asegurar su calidad y rendimiento.
4.  Construye una imagen Docker que contiene una API FastAPI para servir el modelo.
5.  Sube la imagen Docker a Amazon ECR (Elastic Container Registry).
6.  Despliega/actualiza funciones AWS Lambda (`dev` y `prod`) para que usen la nueva imagen de ECR.
7.  Expone las funciones Lambda a través de Amazon API Gateway, proporcionando endpoints públicos para `dev` y `prod`.
8.  Registra cada predicción realizada a través de la API en un archivo de texto en S3 para monitoreo.

## Arquitectura del Sistema

El sistema utiliza los siguientes componentes principales:

*   **GitHub:** Para el control de versiones del código.
*   **GitHub Actions:** Para la orquestación del pipeline de CI/CD.
*   **Python:** Para los scripts de entrenamiento, pruebas y la API.
    *   **Scikit-learn:** Para el entrenamiento del modelo y preprocesamiento.
    *   **ONNX & ONNXRuntime:** Para la serialización y ejecución del modelo.
    *   **FastAPI:** Para construir la API de predicción.
    *   **Boto3:** Para interactuar con servicios de AWS.
    *   **NumPy:** Para manipulación de datos numéricos en la API.
*   **Docker:** Para empaquetar la aplicación FastAPI.
*   **Amazon Web Services (AWS) (Región: `us-east-2`):**
    *   **S3 (Simple Storage Service):**
        *   Bucket: `mlops-winequality-9102`
        *   Almacenamiento del modelo ONNX (`wine_quality_model.onnx`).
        *   Almacenamiento del preprocesador (`wine_quality_scaler.joblib`).
        *   Almacenamiento de datos de prueba (`test_data/test_features.csv`, `test_data/test_labels.csv`).
        *   Almacenamiento de logs de predicciones (`predictions/wine_quality_dev.txt`, `predictions/wine_quality_prod.txt`).
    *   **ECR (Elastic Container Registry):**
        *   Repositorio: `mlops-proj` (URI: `913513915525.dkr.ecr.us-east-2.amazonaws.com/mlops-proj`)
    *   **Lambda:**
        *   Función Dev: `wineQualityApi-dev` (ARN: `arn:aws:lambda:us-east-2:913513915525:function:wineQualityApi-dev`)
        *   Función Prod: `wineQualityApi-prod` (ARN: `arn:aws:lambda:us-east-2:913513915525:function:wineQualityApi-prod`)
    *   **API Gateway:**
        *   API Dev: `WineQualityDevAPI` (URL Invocación: `https://lxs9zt3gmb.execute-api.us-east-2.amazonaws.com/dev`)
        *   API Prod: `WineQualityProdAPI` (URL Invocación: `https://r0wtt8yyz2.execute-api.us-east-2.amazonaws.com/prod`)
    *   **IAM (Identity and Access Management):** Roles y políticas para gestionar permisos.
        *   Usuario para GitHub Actions: `mlops-user`
        *   Roles de Lambda: `lambda-wine-quality-api-dev-role`, `lambda-wine-quality-api-prod-role`
    *   **CloudWatch Logs:** Para el logging de las funciones Lambda y la depuración.

## Flujo de Trabajo de CI/CD

El pipeline de GitHub Actions (`.github/workflows/main.yml`) se activa con un `push` a las ramas `dev` o `prod`.

1.  **Job `train_and_test`:**
    *   Configura el entorno y las credenciales de AWS.
    *   Ejecuta `scripts/train.py`: Entrena/simula un nuevo modelo y sube `wine_quality_model.onnx` y `wine_quality_scaler.joblib` a S3.
    *   Ejecuta `scripts/model_tester.py`: Descarga los artefactos y datos de prueba de S3, realiza pruebas de respuesta y métricas. Si falla, el pipeline se detiene.
2.  **Job `build_and_deploy` (si `test_model` es exitoso):**
    *   Configura credenciales y se loguea a ECR.
    *   Determina el entorno (`dev` o `prod`) basado en la rama.
    *   Construye la imagen Docker (`app/Dockerfile`). La API en la imagen descarga el modelo/scaler de S3 al iniciar.
    *   Etiqueta la imagen con `latest-{env}` y `{commit-sha}-{env}`.
    *   Sube la imagen a ECR.
    *   Actualiza la función AWS Lambda correspondiente (`wineQualityApi-dev` o `wineQualityApi-prod`) para usar la nueva imagen (`latest-{env}`).

## Estructura del Repositorio

## Configuración y Despliegue

### Prerrequisitos
*   Cuenta de AWS.
*   AWS CLI configurada localmente.
*   Docker instalado localmente.
*   Git.

### Configuración de AWS Manual
La siguiente infraestructura de AWS se configuró manualmente en la región `us-east-2`:
*   Bucket S3: `mlops-winequality-9102` (con carpeta `test_data/` y sus archivos).
*   Repositorio ECR: `mlops-proj`.
*   Roles IAM: `mlops-user` (para GitHub Actions), `lambda-wine-quality-api-dev-role`, `lambda-wine-quality-api-prod-role`.
*   Políticas IAM personalizadas: `S3AccessForWineQualityMLOps` (para `mlops-user`), `LambdaWineQualityS3Access-Dev`, `LambdaWineQualityS3Access-Prod`, `GitHubActionsLambdaUpdateAccess` (para `mlops-user`).
*   Funciones Lambda: `wineQualityApi-dev`, `wineQualityApi-prod` (configuradas para imagen de ECR, con roles, variables de entorno, memoria de 2048MB y timeout de 1 min 30 seg).
*   API Gateways: `WineQualityDevAPI` (etapa `dev`), `WineQualityProdAPI` (etapa `prod`).

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

Endpoints (reemplazar `{URL_BASE_API}` con la URL de invocación de API Gateway):

**Endpoint Raíz (Health Check):**
*   `GET {URL_BASE_API}/`
*   Respuesta: `{"message": "Welcome to the Wine Quality Prediction API (dev/prod). Model and scaler are loaded."}`

**Endpoint de Predicción:**
*   `POST {URL_BASE_API}/predict`
*   Cuerpo (JSON): (Ver ejemplo en `app/main.py` o pruebas anteriores)
*   Respuesta (200 OK): `{"predicted_quality_class": X}`

Cada predicción se registra en `s3://mlops-winequality-9102/predictions/wine_quality_{ENVIRONMENT_NAME}.txt`.

## Desafíos y Consideraciones Importantes

### Cold Starts y Timeouts en AWS Lambda
El desafío más significativo encontrado en este proyecto es el manejo de los "cold starts" para las funciones AWS Lambda que utilizan imágenes de contenedor. La imagen Docker para esta aplicación, incluso después de optimizaciones (como la eliminación de `pandas`), tiene un tamaño aproximado de 556MB debido a las dependencias de Machine Learning (`scikit-learn`, `onnxruntime`, `numpy`).

AWS Lambda tiene un límite estricto de **10 segundos para la fase de inicialización (`init`)** de una nueva instancia. Esta fase incluye la descarga de la imagen desde ECR, su descompresión y la preparación del entorno de ejecución de Python. Para imágenes grandes, este proceso puede exceder consistentemente los 10 segundos, resultando en un `init timeout`.

**Observaciones:**
*   Los logs de CloudWatch muestran repetidos `INIT_REPORT Init Duration: ~10000 ms Phase: init Status: timeout`.
*   Los `print` de diagnóstico a nivel de módulo y al inicio de la función `startup_event` en `app/main.py` no aparecen antes de este `init timeout`, indicando que el tiempo se consume principalmente en la preparación de la imagen por parte de Lambda.
*   En algunos reintentos o con instancias "semi-calientes", la aplicación FastAPI *logra* iniciarse (como lo indican los logs "Application startup complete" y "Uvicorn running").
*   Sin embargo, incluso cuando la aplicación se inicia, la primera invocación (que incluye la ejecución de `startup_event` con descargas S3 y carga de modelos) puede exceder el tiempo de espera de API Gateway (29s) o el tiempo de espera de la propia Lambda si es muy largo. Se configuró la Lambda con 2048MB de RAM y un timeout de 1 minuto 30 segundos para mitigar esto en la fase de invocación.

**Mitigación para la Demostración y Uso:**
1.  **"Calentar" la Lambda:** Antes de una demostración o para pruebas consistentes, es necesario invocar el endpoint de API Gateway varias veces. Las primeras peticiones pueden fallar o dar timeout. Eventualmente, una instancia de Lambda se inicializará completamente.
2.  **Invocaciones Posteriores:** Una vez que una instancia está "caliente", las invocaciones subsiguientes a esa misma instancia son rápidas y funcionales.
3.  **Documentación:** Este comportamiento es una limitación conocida y debe ser considerado.

Este desafío subraya la importancia de la optimización del tamaño de la imagen y la gestión de cold starts en arquitecturas serverless para ML.

## Futuras Mejoras
*   **Optimización Agresiva del Tamaño de la Imagen:** Investigar técnicas más avanzadas para reducir el tamaño de `scikit-learn` y `onnxruntime` o explorar alternativas más ligeras si la funcionalidad lo permite.
*   **AWS Lambda Provisioned Concurrency:** Para entornos de producción donde los cold starts no son aceptables, configurar concurrencia aprovisionada (conlleva costos adicionales).
*   **Lambda Layers:** Para dependencias comunes y grandes, podrían empaquetarse en capas Lambda para mejorar los tiempos de inicio (aunque con imágenes de contenedor, el beneficio puede ser menor).
*   **Infraestructura como Código (IaC):** Utilizar herramientas como AWS CDK, Terraform o CloudFormation para definir y gestionar todos los recursos de AWS de forma reproducible.
*   **Monitoreo Avanzado:** Implementar un monitoreo más detallado del rendimiento del modelo, deriva de datos y logs de la aplicación.
*   **Seguridad Mejorada:** Refinar los permisos IAM al mínimo estrictamente necesario, implementar autenticación en API Gateway si es necesario.
*   **Pipeline de Reentrenamiento:** Automatizar el reentrenamiento del modelo basado en triggers (nuevos datos, degradación del rendimiento).

---