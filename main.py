from typing import Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from recommender.recommender import aesthetic_simmilarity_recommendation, affinity_recommendation

MODEL_FILENAME = '???.pkl'

# model = load(MODEL_FILENAME)
app = FastAPI()

@app.get("/ping")
async def ping():
    return { "status": "ok" }

@app.get("/recommendation/{item_id}")
async def get_recommendation(item_id: str, user_address: Union[str, None] = None, type: Union[str, None] = None):
    if type == 'similarity':
        headers = { 'Access-Control-Allow-Origin' : '*' }
        try:
            return JSONResponse(content=aesthetic_simmilarity_recommendation(item_id), headers=headers)
        except:
            return JSONResponse(content={ "message": "No recommendations found" }, status_code=404, headers=headers)
    elif type == 'othersBought':
        headers = { 'Access-Control-Allow-Origin' : '*' }
        try:
            return JSONResponse(content=affinity_recommendation(item_id), headers=headers)
        except:
            return JSONResponse(content={ "message": "No recommendations found" }, status_code=404, headers=headers)
