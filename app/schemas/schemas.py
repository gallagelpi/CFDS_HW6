from pydantic import BaseModel

# ============================================================
# 🧠 PREDICTION SCHEMAS
# ============================================================
# Schemas of predict endpoint

class PredictRequest(BaseModel):
    age: float
    bmi: float
    heart_rate_apache: float
    temp_apache: float
    map_apache: float
    resprate_apache: float
    glucose_apache: float
    creatinine_apache: float
    wbc_apache: float
    gender: str
    ethnicity: str



class PredictResponse(BaseModel):
    model_used: str
    probability_diabetes: float
    predicted_class: int


# ============================================================
# 🧩 TRAINING SCHEMAS
# ============================================================
#Schemas of train_model endpoint

class TrainRequest(BaseModel):
    penalty: str = "l2"
    max_iter: int = 100


class TrainResponse(BaseModel):
    message: str
    model_path: str
    penalty: str
    max_iter: int
