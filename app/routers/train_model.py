from fastapi import APIRouter, HTTPException
from app.schemas.schemas import TrainRequest, TrainResponse
from app.services.train_model import train

router = APIRouter(prefix="/train", tags=["Training"])

@router.post("/", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a Logistic Regression model and save it in a timestamped folder.
    """
    try:
        result = await train(request.penalty, request.max_iter)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
