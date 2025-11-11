from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PredictRequest, PredictResponse
from app.services.predict import predict_datapoint

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictResponse)
from app.schemas.schemas import PredictRequest
from app.services.predict import predict_datapoint

router = APIRouter()

@router.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict diabetes likelihood for a single ICU datapoint.
    """

    try:
        result = await predict_datapoint(
            age=request.age,
            bmi=request.bmi,
            heart_rate_apache=request.heart_rate_apache,
            temp_apache=request.temp_apache,
            map_apache=request.map_apache,
            resprate_apache=request.resprate_apache,
            glucose_apache=request.glucose_apache,
            creatinine_apache=request.creatinine_apache,
            wbc_apache=request.wbc_apache,
            gender=request.gender,
            ethnicity=request.ethnicity
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
