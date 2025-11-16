<<<<<<< HEAD
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib, os, uvicorn
import traceback
from datetime import datetime

# Load Model 
loaded_model = joblib.load('component/trained_random_forest_model.pkl')

# Load Encoders 
save_directory = 'component/encoders'
loaded_encoders = {}
for filename in os.listdir(save_directory):
    if filename.endswith(".joblib"):
        feature_name = filename.replace("_encoder.joblib", "")
        full_path = os.path.join(save_directory, filename)
        loaded_encoders[feature_name] = joblib.load(full_path)

#  Define FastAPI App 
app = FastAPI(title="Customer Churn Prediction API")

# Define Input Schema
class TransactionData(BaseModel):
    CustomerID: int
    Tenure: int
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: int
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: int
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: int
    CouponUsed: int
    OrderCount: int
    DaySinceLastOrder: int
    CashbackAmount: float

# Prediction Function 
def predictChurn(Data):
    try:
        df = pd.DataFrame([Data])
        df = df.drop(columns=['CustomerID'])
        
        # Encode categorical features
        for col, encoder in loaded_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        result = int(loaded_model.predict(df)[0])
        prediction = "Churn" if result == 1 else "Retained"

        # Log result to file
        with open("api_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} | SUCCESS | {Data} | Prediction: {prediction} ({result})\n")

        return result, prediction

    except Exception as e:
        # Log error
        with open("api_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} | ERROR | {Data} | {traceback.format_exc()}\n")

        # Raise HTTPException for FastAPI
        raise HTTPException(status_code=500, detail=str(e))

# Define API Endpoint 
@app.post("/predict")
def predict(data: TransactionData):
    data_dict = data.model_dump()
    result, prediction = predictChurn(data_dict)
    return {
        "result": result,
        "prediction": prediction
    }

# ========== Run Server ==========
if __name__ == "__main__":
    print("🚀 Starting API server on http://127.0.0.1:8000")
    uvicorn.run("predictChurnAPI:app", host="127.0.0.1", port=8000, reload=True)


"""
HTTP Methode : POST
URL : http://127.0.0.1:8000
Body(Json) : {
    "CustomerID": 12345,
    "Tenure": 1,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 2,
    "WarehouseToHome": 15,
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Male",
    "HourSpendOnApp": 3,
    "NumberOfDeviceRegistered": 5,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 4,
    "MaritalStatus": "Single",
    "NumberOfAddress": 1,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12,
    "CouponUsed": 2,
    "OrderCount": 5,
    "DaySinceLastOrder": 3,
    "CashbackAmount": 150.5
}
=======
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib, os, uvicorn
import traceback
from datetime import datetime

# Load Model 
loaded_model = joblib.load('component/trained_random_forest_model.pkl')

# Load Encoders 
save_directory = 'component/encoders'
loaded_encoders = {}
for filename in os.listdir(save_directory):
    if filename.endswith(".joblib"):
        feature_name = filename.replace("_encoder.joblib", "")
        full_path = os.path.join(save_directory, filename)
        loaded_encoders[feature_name] = joblib.load(full_path)

#  Define FastAPI App 
app = FastAPI(title="Customer Churn Prediction API")

# Define Input Schema
class TransactionData(BaseModel):
    CustomerID: int
    Tenure: int
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: int
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: int
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: int
    CouponUsed: int
    OrderCount: int
    DaySinceLastOrder: int
    CashbackAmount: float

# Prediction Function 
def predictChurn(Data):
    try:
        df = pd.DataFrame([Data])
        df = df.drop(columns=['CustomerID'])
        
        # Encode categorical features
        for col, encoder in loaded_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        result = int(loaded_model.predict(df)[0])
        prediction = "Churn" if result == 1 else "Retained"

        # Log result to file
        with open("api_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} | SUCCESS | {Data} | Prediction: {prediction} ({result})\n")

        return result, prediction

    except Exception as e:
        # Log error
        with open("api_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} | ERROR | {Data} | {traceback.format_exc()}\n")

        # Raise HTTPException for FastAPI
        raise HTTPException(status_code=500, detail=str(e))

# Define API Endpoint 
@app.post("/predict")
def predict(data: TransactionData):
    data_dict = data.model_dump()
    result, prediction = predictChurn(data_dict)
    return {
        "result": result,
        "prediction": prediction
    }

# ========== Run Server ==========
if __name__ == "__main__":
    print("🚀 Starting API server on http://127.0.0.1:8000")
    uvicorn.run("predictChurnAPI:app", host="127.0.0.1", port=8000, reload=True)


"""
HTTP Methode : POST
URL : http://127.0.0.1:8000
Body(Json) : {
    "CustomerID": 12345,
    "Tenure": 1,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 2,
    "WarehouseToHome": 15,
    "PreferredPaymentMode": "Credit Card",
    "Gender": "Male",
    "HourSpendOnApp": 3,
    "NumberOfDeviceRegistered": 5,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 4,
    "MaritalStatus": "Single",
    "NumberOfAddress": 1,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12,
    "CouponUsed": 2,
    "OrderCount": 5,
    "DaySinceLastOrder": 3,
    "CashbackAmount": 150.5
}
>>>>>>> 1c0a3f77d2c037c50d7719c5dbae0fe84ed65b52
"""