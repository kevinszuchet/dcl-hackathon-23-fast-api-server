from typing import Union
from fastapi import FastAPI
from joblib import load

MODEL_FILENAME = '???.pkl'

# model = load(MODEL_FILENAME)
app = FastAPI()

@app.get("/recommendation/{item_id}")
async def get_recommendation(item_id: str, user_address: Union[str, None] = None, type: Union[str, None] = None):
    # type: 'similarity' or 'othersBought'
    return {"item_id":item_id,"user_address":user_address,"type":type}
    # inputs = {"item_id": item_id, "user_address": user_address}
    # [prediction] = model.predict(inputs)
    # print("Prediction:", prediction)
    # return str(prediction)
