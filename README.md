# Proyecto Whisper LARA

Proyecto para transcripción de audio utilizando el modelo Whisper con seguimiento en MLflow.

## Estructura del proyecto

```
├── src/                     # Código fuente
├── tests/                   # Pruebas unitarias
├── .github/workflows/       # Configuración de CI
├── train_whisper.py         # Script de entrenamiento
├── evaluate_whisper.py      # Script de evaluación
├── Makefile                 # Automatización de tareas
├── requirements.txt         # Dependencias
└── .pre-commit-config.yaml  # Configuración de pre-commit
```

## Instalación inicial

```bash
# Configurar el entorno
make setup

# Activar el entorno virtual
source venv/bin/activate

# Instalar dependencias
make install
```

## Flujo de desarrollo

1. Crear una rama feature: `git checkout -b feature/nombre_tarea`
2. Desarrollar tu código
3. Ejecutar pruebas: `make test`
4. Iniciar MLflow: `make mlflow` (en una terminal separada)
5. Entrenar modelo: `make train`
6. Evaluar modelo: `make evaluate`
7. Commit, push y crear un PR a development

## Estructura de ramas

- `master`: Código estable para producción
- `development`: Integración de features
- `feature/nombre`: Desarrollo individual

## CI/CD

Las pruebas, entrenamiento y evaluación se ejecutan automáticamente en GitHub Actions.