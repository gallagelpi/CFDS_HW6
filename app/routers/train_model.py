from fastapi import FastAPI
from app.services import train_model

app = FastAPI()

@app.post("/train_model")
async def train(penalty: str = 'l2', max_iter: int = 100):
    return train_model.train(penalty, max_iter)