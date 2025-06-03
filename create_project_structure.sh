#!/bin/bash

mkdir -p .github/workflows
mkdir -p app
mkdir -p scripts
mkdir -p notebooks

touch .github/workflows/main.yml
touch app/main.py
touch app/Dockerfile
touch app/requirements.txt
touch scripts/train.py
touch notebooks/wine_exploration.ipynb
touch .gitignore
touch requirements_dev.txt
touch README.md

echo "Estructura de carpetas y archivos iniciales creada."
