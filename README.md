# URL Spam Detection with NLP and SVM

![URL Spam Detection project banner](assets/project-banner.png)

**Language / Idioma:** English | [Español](README.es.md)

This project classifies URLs as spam or not spam using only the link text. It implements the complete assignment workflow: data loading, URL preprocessing, train/test split, baseline SVM, hyperparameter optimization and model persistence.

**Español:** Este proyecto detecta enlaces spam usando solamente la URL. Incluye carga de datos, preprocesamiento NLP, division train/test, SVM base, optimizacion de hiperparametros y guardado del modelo final.

## What the pipeline does

1. Loads `data/raw/url_spam.csv`.
2. Cleans duplicate rows and validates the required columns.
3. Splits the dataset into stratified train and test files.
4. Tokenizes URLs by punctuation and URL components.
5. Removes stopwords and applies lightweight lemmatization.
6. Builds TF-IDF text features and engineered URL statistics.
7. Trains a default SVM baseline.
8. Optimizes an SVM with `GridSearchCV`.
9. Saves the best model and metrics under `models/`.

## Project Structure

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
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/app.py
```

The model artifact is saved to:

```text
models/url_spam_svm_pipeline.joblib
```

The full evaluation report is saved to:

```text
models/url_spam_svm_metrics.json
```

## Predict with the Saved Model

```python
import joblib
import sys

sys.path.append("src")

model = joblib.load("models/url_spam_svm_pipeline.joblib")
prediction = model.predict(["https://briefingday.us8.list-manage.com/unsubscribe"])
print(bool(prediction[0]))
```

`True` means spam and `False` means not spam.
