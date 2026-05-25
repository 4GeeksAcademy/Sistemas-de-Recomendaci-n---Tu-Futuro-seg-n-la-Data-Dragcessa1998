# Recommendation Systems - Your Future Through Data

![Project banner](assets/project-banner.png)

**Language / Idioma:** English | [Español](README.es.md)

This project uses the **Adult Income Dataset** to train a supervised classifier that estimates whether an adult profile is likely to earn more than **50K USD per year**. The prediction layer feeds an interpretive recommender that suggests actionable education and career paths to improve the estimated probability.

**Español:** Este proyecto usa el Adult Income Dataset para estimar si un perfil adulto podria superar los 50,000 USD anuales y recomendar trayectorias de educacion u ocupacion para mejorar esa probabilidad.

## Goals

- Explore census income data.
- Clean null or malformed values such as `?`.
- Transform categorical variables with One-Hot Encoding.
- Normalize numerical variables.
- Train a supervised classification model.
- Define and build an interpretive recommendation system.
- Test recommendations with simulated user profiles.

## Recommendation Problem

- **What is recommended:** actionable improvement paths, such as increasing education level, exploring high-income occupations, changing workclass and adjusting weekly hours.
- **Who is the user:** an adult person represented by a demographic and socioeconomic profile.
- **Profile variables:** age, education, occupation, marital status, relationship, hours per week, workclass, country, sex, race and capital variables.
- **Approach:** hybrid recommender. It combines a supervised `>50K` probability model with content-based filtering using similar high-income profiles.

## Workflow

1. Load `data/raw/adult-census-income.csv`.
2. Clean columns, duplicates and `?` values.
3. Create binary target `income_high`.
4. Build a stratified train/test split.
5. Preprocess data:
   - median imputation for numerical features;
   - `StandardScaler` for numerical features;
   - `Unknown` imputation for categorical features;
   - `OneHotEncoder` for categorical features.
6. Train a baseline classifier.
7. Optimize it with `GridSearchCV`.
8. Build the hybrid recommender.
9. Test simulated user profiles.
10. Save model and result artifacts.

## Structure

```text
.
├── assets/project-banner.png
├── data/
│   ├── raw/adult-census-income.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── adult_income_recommender.joblib
│   ├── adult_income_metrics.json
│   └── sample_recommendations.json
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

The script creates:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/adult_income_recommender.joblib`
- `models/adult_income_metrics.json`
- `models/sample_recommendations.json`

## Use the Recommender

```python
import joblib
import sys

sys.path.append("src")

artifact = joblib.load("models/adult_income_recommender.joblib")
recommender = artifact["recommender"]

user_profile = {
    "age": 25,
    "workclass": "Private",
    "fnlwgt": 180000,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 25,
    "native_country": "United-States",
}

result = recommender.recommend(user_profile)
print(result.base_probability)
print(result.recommendations)
```
