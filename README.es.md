# Sistema de deteccion de URLs spam

![Banner del proyecto de deteccion de URLs spam](assets/project-banner.png)

**Idioma / Language:** [English](README.md) | Español

Proyecto de NLP para clasificar automaticamente si una URL contiene spam. La solucion parte de `url_spam.csv`, transforma cada enlace en tokens utiles para aprendizaje automatico y entrena un SVM con una version base y otra optimizada mediante grid search.

**English:** This project detects spam links using only the URL. It includes data loading, NLP preprocessing, train/test split, baseline SVM, hyperparameter optimization and final model persistence.

## Objetivo

Detectar enlaces spam usando solamente la URL, sin acceder al contenido de la pagina. Este enfoque es util cuando se necesita una primera capa rapida de filtrado para emails, newsletters, formularios o sistemas de moderacion.

## Flujo de trabajo

1. Carga y validacion del dataset `data/raw/url_spam.csv`.
2. Limpieza de duplicados y division estratificada en train/test.
3. Preprocesamiento NLP especifico para URLs:
   - segmentacion por signos de puntuacion;
   - eliminacion de stopwords;
   - normalizacion y lematizacion ligera;
   - TF-IDF con unigramas y bigramas.
4. Extraccion de variables numericas de URL:
   - longitud;
   - cantidad de digitos;
   - cantidad de caracteres especiales;
   - puntos, guiones y barras;
   - presencia de query string;
   - uso de HTTPS;
   - numero aproximado de subdominios.
5. Entrenamiento de un SVM base con parametros por defecto.
6. Optimizacion con `GridSearchCV`.
7. Guardado del mejor pipeline completo en `models/`.

## Estructura

```text
.
├── data/
│   ├── raw/url_spam.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── url_spam_svm_pipeline.joblib
│   └── url_spam_svm_metrics.json
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
├── requirements.txt
└── README.es.md
```

## Instalacion

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

Desde la raiz del repositorio:

```bash
python src/app.py
```

El script genera:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/url_spam_svm_pipeline.joblib`
- `models/url_spam_svm_metrics.json`

## Resultados

Las metricas se guardan automaticamente en `models/url_spam_svm_metrics.json`. El archivo incluye:

- distribucion del dataset;
- metricas del SVM base;
- mejores hiperparametros del SVM optimizado;
- accuracy, precision, recall y F1 para la clase spam;
- matriz de confusion;
- rutas de los artefactos generados.

## Uso del modelo

```python
import joblib
import sys

sys.path.append("src")

model = joblib.load("models/url_spam_svm_pipeline.joblib")
prediction = model.predict(["https://briefingday.us8.list-manage.com/unsubscribe"])
print(bool(prediction[0]))
```

`True` significa spam y `False` significa no spam.
