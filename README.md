# SVM Classifier App

Aplicación web interactiva para demostrar el uso de Máquinas de Vectores de Soporte (SVM) en tareas de clasificación.

## Estructura del Proyecto

```
svm-app/
│
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias del proyecto
│
├── models/
│   ├── __init__.py
│   └── svm_classifier.py          # Lógica del modelo SVM
│
├── utils/
│   ├── __init__.py
│   ├── data_processing.py         # Procesamiento de datos
│   └── visualization.py           # Visualizaciones
│
├── data/
│   └── iris_sample.csv           # Dataset de ejemplo
│
└── assets/                        # Recursos adicionales
```

## Instalación

1. Activar el entorno virtual:
```bash
.\venv\Scripts\Activate.ps1
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecutar la Aplicación

```bash
streamlit run app.py
```

## Características

- Carga de datasets personalizados (CSV)
- Selección de features y target
- Configuración de hiperparámetros del SVM
- Múltiples kernels (linear, rbf, poly, sigmoid)
- Visualización de métricas de rendimiento
- Matriz de confusión
- Frontera de decisión
- Curva ROC (para clasificación binaria)
