# Usar una imagen base oficial de Python.
# python:3.9-slim es una buena opción por ser relativamente ligera.
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copiar solo el archivo de requerimientos primero para aprovechar el caché de Docker.
# Si requirements.txt no cambia, esta capa no se reconstruirá innecesariamente.
COPY ./requirements.txt /app/requirements.txt

# Instalar las dependencias de Python.
# --no-cache-dir para no guardar el caché de pip y mantener la imagen más pequeña.
# --upgrade pip para asegurar que pip esté actualizado.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copiar el resto del código de la aplicación (main.py y cualquier otro archivo en app/)
# al directorio de trabajo /app en el contenedor.
COPY . /app/

# Comando para ejecutar la aplicación FastAPI con Uvicorn cuando el contenedor se inicie.
# Escuchará en todas las interfaces (0.0.0.0) en el puerto 8000.
# El puerto 8000 es el que Lambda espera por defecto para imágenes de contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]