import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from pathlib import Path


async def train(penalty: str, max_iter: int):
    """
    Trains a Logistic Regression model using the ICU dataset and saves
    it into a timestamped folder inside ../models.
    """
    if penalty not in ["l2", "none"]:
        raise ValueError("Invalid penalty type. Choose from 'l2' or 'none'.")

    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    # Load dataset
    df = pd.read_csv("data/sample_diabetes_mellitus_data.csv")

    # Define target and selected features
    target = "diabetes_mellitus"
    numeric_features = [
        "age", "bmi", "heart_rate_apache", "temp_apache",
        "map_apache", "resprate_apache", "glucose_apache",
        "creatinine_apache", "wbc_apache"
    ]
    categorical_features = ["gender", "ethnicity"]

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Split data
    X = df[numeric_features + categorical_features]
    y = df[target]

    # Build preprocessing pipeline
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Build model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            penalty=penalty,
            max_iter=max_iter,
            random_state=42
        ))
    ])

    # Fit model
    model.fit(X, y)

    # Save pipeline
    model_path = "app/logistic_regression_model.pkl"
    joblib.dump(model, model_path)

    # Return response
    return "Model trained successfully with imputation and encoding."
   