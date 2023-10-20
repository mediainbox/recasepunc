from fastapi import FastAPI
from recasepunc import WordpieceTokenizer
from recasor import Recasor
from pydantic import BaseModel
import uvicorn
app = FastAPI()

recasor = None

@app.get("/")
async def root():
    return {"status": "ok"}

# a function that runs on startup
@app.on_event("startup")
async def startup_event():
    global recasor
    recasor = Recasor()
    print("Starting up...")


class PredictPost(BaseModel):
    text: str

# create a predict endpoint that receive a post for a text and return a prediction
# pass the text through the body post
@app.post("/predict")
async def predict(req: PredictPost):
    data = recasor.predict(req.text)
    return {"prediction": data}


if __name__ == "__main__":
    uvicorn.run(app, host="", port=8000)
