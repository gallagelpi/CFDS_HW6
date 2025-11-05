from sklearn import LogisticRegression
import pandas as pd
import joblib

async def train(penalty: str, max_iter: int):

    df = pd.read_csv('../data/sample_diabetes_mellitus_data.csv')
    model = LogisticRegression(penalty=penalty, max_iter=max_iter, random_state=42)

    y = df['Diabetes_Mellitus']
    X = df[['bmi', 'age', 'hr', 'systolic_bp', 'diastolic_bp']]

    model.fit(X, y)

    #Create a pickle file to save the model
    with open('../app/logistic_regression_model.pkl', 'wb') as f:
        joblib.dump()(model, f)

    return {"messange": f"Model trained with penalty={penalty} and max_iter={max_iter}"}