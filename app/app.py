from fastapi import FastAPI
from app.routers import train_model, predict

app = FastAPI(title="Diabetes Prediction API")

app.include_router(train_model.router)
app.include_router(predict.router)
app.include_router(predict.router)
