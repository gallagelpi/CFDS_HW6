from fastapi import APIRouter
from app.schemas.schemas import TrainRequest
from app.services.train_model import train

router = APIRouter()

@router.post("/train")
async def train_model(request: TrainRequest):
    """
    Train a Logistic Regression model and save it in a timestamped folder.
    """
    try:
        return await train(request.penalty, request.max_iter)
    except ValueError as e:
        return {"error": str(e)}