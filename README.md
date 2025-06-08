
## Configuración y Despliegue

### Prerrequisitos
*   Cuenta de AWS.
*   AWS CLI configurada localmente.
*   Docker instalado localmente.
*   Git.
*   Archivo de clave `.pem` para acceso SSH a la instancia EC2 de `dev`.

### Configuración de AWS Manual
La siguiente infraestructura de AWS se configuró manualmente en la región `us-east-2`:
*   Bucket S3: `mlops-winequality-9102` (con `wine_quality_model.onnx`, `wine_quality_scaler.joblib`, y carpeta `test_data/` con sus archivos).
*   Repositorio ECR: `mlops-proj`.
*   Instancia EC2 para `dev` (con un rol IAM `ec2-mlops-api-role` que permite acceso a ECR y S3).
*   Roles IAM: `mlops-user` (para GitHub Actions), `lambda-wine-quality-api-prod-role`.
*   Políticas IAM personalizadas (ej. `S3AccessForWineQualityMLOps`, `LambdaWineQualityS3Access-Prod`, `GitHubActionsLambdaUpdateAccess`, `EC2InstanceMLOpsAPIAccess`).
*   Función Lambda: `wineQualityApi-prod` (configurada para imagen de ECR, con rol, variables de entorno, memoria de 2048MB y timeout de 1 min 30 seg).
*   API Gateway: `WineQualityProdAPI` (etapa `prod`).

### Variables de Entorno y Secrets de GitHub
**Secrets del Repositorio GitHub (Settings > Secrets and variables > Actions):**
*   `AWS_ACCESS_KEY_ID`: `AKIA5JMNMESCT2ULRP45`
*   `AWS_SECRET_ACCESS_KEY`: (La secret key correspondiente)
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`
*   `DEV_EC2_HOST`: `18.216.171.109` (IP de la instancia EC2 para `dev`)
*   `DEV_EC2_USERNAME`: `ubuntu` (o el usuario SSH de la EC2 de `dev`)
*   `DEV_EC2_SSH_KEY`: (La clave privada SSH para la EC2 de `dev`)

**Variables de Entorno para la Función Lambda `prod` (configuradas en la consola de Lambda):**
*   `S3_BUCKET_NAME`: `mlops-winequality-9102`
*   `S3_MODEL_KEY_ONNX`: `wine_quality_model.onnx`
*   `S3_SCALER_KEY`: `wine_quality_scaler.joblib`
*   `PREDICTION_LOG_KEY_PREFIX`: `predictions/wine_quality`
*   `ENVIRONMENT_NAME`: `prod`

**Variables de Entorno para el Contenedor en EC2 `dev` (pasadas por el `docker run` en el pipeline):**
*   Similares a las de Lambda, con `ENVIRONMENT_NAME="dev-ec2"`.

## Uso de la API

**Entorno `dev` (EC2):**
*   Raíz (Health Check): `GET http://18.216.171.109:8000/`
*   Predicción: `POST http://18.216.171.109:8000/predict`

**Entorno `prod` (Lambda):**
*   Raíz (Health Check): `GET https://r0wtt8yyz2.execute-api.us-east-2.amazonaws.com/prod/`
*   Predicción: `POST https://r0wtt8yyz2.execute-api.us-east-2.amazonaws.com/prod/predict`

Cuerpo de la solicitud `POST /predict` (JSON): (Ver ejemplo en `app/main.py` o pruebas anteriores).
Respuesta Exitosa (200 OK): `{"predicted_quality_class": X}`.
Logs de predicción: `s3://mlops-winequality-9102/predictions/wine_quality_{ENVIRONMENT_NAME}.txt`.

## Pruebas Implementadas
El pipeline de CI/CD incluye:
1.  **Pruebas Unitarias (`tests/test_example_utils.py`):** Ejecutadas por `pytest` para verificar lógica básica.
2.  **Pruebas de Integración del Modelo (`scripts/model_tester.py`):** Descarga artefactos de S3 y prueba la respuesta y métrica (accuracy) del modelo.

Ambos conjuntos deben pasar para proceder al despliegue.

## Desafíos, Consideraciones Importantes y Estrategias de Mitigación

### Cold Starts, Timeouts en AWS Lambda (`prod`) y Límites de Concurrencia
El desafío más significativo para el entorno `prod` (AWS Lambda) es el manejo de "cold starts". La imagen Docker de la aplicación (~556MB) es considerable debido a las dependencias de Machine Learning.

**1. `init timeout` de Lambda:** AWS Lambda tiene un límite de 10 segundos para la fase de inicialización (`init`) de una instancia fría (descarga/descompresión de imagen, inicio del runtime). Para imágenes grandes, este límite puede excederse consistentemente, como se observa en los logs de CloudWatch (`INIT_REPORT ... Status: timeout`). Esto ocurre antes de que el código de la aplicación en `app/main.py` se ejecute de forma significativa.

**2. Timeouts de Invocación:** En reintentos o instancias "semi-calientes" donde la `init` logra pasar, la primera invocación real (que ejecuta `startup_event` con descargas S3 y carga de modelos, más el procesamiento de la petición) puede exceder el tiempo de espera de la Lambda (configurado en 1m 30s) o el límite de 29 segundos de API Gateway, resultando en un timeout para el cliente.

**3. Imposibilidad de Usar Provisioned Concurrency (en el contexto de este proyecto):** Aunque Provisioned Concurrency es una solución para eliminar cold starts, no se implementó debido a los límites de concurrencia predeterminados en la cuenta de AWS utilizada y para mantener el proyecto dentro de la capa gratuita.

**Estrategia de Mitigación para la Demostración (`prod`):**
*   **"Calentar" la Lambda:** Es necesario invocar el endpoint de API Gateway de `prod` varias veces antes de una demostración. Las primeras peticiones pueden fallar. Eventualmente, una instancia se inicializará.
*   **Configuración de Lambda:** `prod` está configurada con 2048MB de RAM y un timeout de 1m 30s para dar margen una vez que la `init` pasa.
*   **Invocaciones Posteriores:** Instancias "calientes" responden rápidamente.

### Entorno `dev` en EC2 como Alternativa Funcional
Debido a los desafíos con Lambda, el entorno `dev` se configuró para desplegarse en una instancia EC2. Este entorno ha demostrado ser funcional y estable para las predicciones y el logging a S3, sirviendo como una demostración robusta de la API.

Este proyecto subraya la importancia de la optimización del tamaño de la imagen y la gestión de cold starts en arquitecturas serverless para ML.

## Futuras Mejoras
*   Optimización Agresiva del Tamaño de la Imagen Docker para Lambda.
*   Implementar AWS Lambda Provisioned Concurrency para `prod` (considerando costos y límites).
*   Infraestructura como Código (IaC) con AWS CDK, Terraform, o CloudFormation.
*   Monitoreo Avanzado del modelo y la aplicación.
*   Refinar la Seguridad (permisos IAM, autenticación API).
*   Pipeline de Reentrenamiento Automatizado.

---