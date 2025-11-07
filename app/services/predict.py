import pandas as pd
import joblib
from pathlib import Path
from app.schemas.schemas import PredictResponse


async def predict_datapoint(
    age: float,
    bmi: float,
    heart_rate_apache: float,
    temp_apache: float,
    map_apache: float,
    resprate_apache: float,
    glucose_apache: float,
    creatinine_apache: float,
    wbc_apache: float,
    gender: str,
    ethnicity: str
) -> PredictResponse:
    """
    Loads the most recent trained Logistic Regression pipeline and
    predicts diabetes probability for a single ICU datapoint.
    """

    # Read the model
    model_path = "app/logistic_regression_model.pkl"

    # Load pipeline (includes preprocessing + model)
    model = joblib.load(model_path)

    if not model:
        raise FileNotFoundError("No trained model loaded.")

    # Prepare input dataframe — same columns used in training
    X_new = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "heart_rate_apache": heart_rate_apache,
        "temp_apache": temp_apache,
        "map_apache": map_apache,
        "resprate_apache": resprate_apache,
        "glucose_apache": glucose_apache,
        "creatinine_apache": creatinine_apache,
        "wbc_apache": wbc_apache,
        "gender": gender,
        "ethnicity": ethnicity
    }])

    # Predict probability and class
    y_pred_prob = float(model.predict_proba(X_new)[0, 1])
    y_pred_class = int(model.predict(X_new)[0])

    # Return structured response
    return PredictResponse(
        model_used=str(model_path),
        probability_diabetes=y_pred_prob,
        predicted_class=y_pred_class
    )