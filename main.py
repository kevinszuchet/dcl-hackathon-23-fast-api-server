from typing import Union
from fastapi import FastAPI
from joblib import load

MODEL_FILENAME = '???.pkl'

# model = load(MODEL_FILENAME)
app = FastAPI()

@app.get("/recommendation")
async def get_recommendation(user_address: Union[str, None] = None, item_id: Union[str, None] = None):
    return {"item_id":item_id,"user_address":user_address}
    # inputs = {"item_id": item_id, "user_address": user_address}
    # [prediction] = model.predict(inputs)
    # print("Prediction:", prediction)
    # return str(prediction)
