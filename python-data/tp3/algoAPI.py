from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import sklearn

app = FastAPI()
model = joblib.load("resultCLF.pkl")

#on fait un classe pour faciliter les appels de donnee ou les 2 models
class Tweet(BaseModel):
    tweet = str

#creation des requetes et des routes aui seront appeler part une URL (localhost:4200/)
@app.get("/")
def read_root():
    return {"Hello": "World"}
  
# recuperation des donnees du model resultCLF et la route URL (localhost:4200/dataTweet)
@app.post("/dataTweet")
def class_response(txt:Tweet): 
    if (txt.tweet) :
        Y_pred = model.predict_proba(vars()[txt.tweet])[0]

        if Y_pred[0] > Y_pred[1] and Y_pred[0]> Y_pred[2]:
            return("hate_speech")
        elif Y_pred[1] > Y_pred[0] and Y_pred[1]> Y_pred[2] :
            return("offensive_language")
        else :
            return("neither")
          
#demmarage du serveur uvicorn
if __name__ == '__main__' :
    uvicorn.run(app,host="127.0.0.1",port="4200")