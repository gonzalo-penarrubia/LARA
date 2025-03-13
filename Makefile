.PHONY: setup install test mlflow train evaluate clean

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip

# Configura el entorno básico
setup:
	$(PIP) install --upgrade pip
	$(PIP) install virtualenv
	virtualenv venv
	@echo "Entorno virtual creado. Actívalo con: source venv/bin/activate"

# Instala las dependencias del proyecto
install:
	$(PIP) install -r requirements.txt
	pre-commit install

# Ejecuta los tests
test:
	pytest tests/ --cov=src

# Inicia el servidor MLflow local
mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Entrena el modelo Whisper
train:
	$(PYTHON) train_whisper.py

# Evalúa el modelo Whisper
evaluate:
	$(PYTHON) evaluate_whisper.py

# Flujo completo: entrena y evalúa
pipeline: mlflow train evaluate

# Limpia archivos temporales
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf mlruns/
	rm -rf mlflow.db