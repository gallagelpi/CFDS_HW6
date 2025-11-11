import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from pathlib import Path
from app.schemas.schemas import TrainResponse


async def train(penalty: str, max_iter: int) -> TrainResponse:
    """
    Trains a Logistic Regression model using the ICU dataset and saves
    it into a timestamped folder inside ../models.
    """

    # Validate parameters
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

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = Path(f"models/{timestamp}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    model_path = folder_path / "logistic_regression_model.pkl"
    joblib.dump(model, model_path)

    # Return structured response
    return TrainResponse(
        message="Model trained successfully with imputation and encoding.",
        model_path=str(model_path),
        penalty=penalty,
        max_iter=max_iter
    )
