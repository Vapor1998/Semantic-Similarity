# pip install -U sentence-transformers
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from Model import Model


class request_body(BaseModel):
    text1 : str
    text2 : str


app = FastAPI()
# Defining path operation for root endpoint
# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Pass Two Texts'}


@app.post('/predict')
def predict(data: request_body):
    test_data = [[
            data.text1,
            data.text2
    ]]

    model = Model()
    pred = model.predict(test_data[0][0], test_data[0][1])


    return {'similarity_score' : round(pred, 2)}


if __name__ == '__main__':
    # app.debug = True
    uvicorn.run(app, host='127.0.0.1', port='8080')



# from sentence_transformers import SentenceTransformer, util
# sentences = ["I'm happy", "I'm full of happiness"]
#
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#
# #Compute embedding for both lists
# embedding_1= model.encode(sentences[0], convert_to_tensor=True)
# embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
#
# print(util.pytorch_cos_sim(embedding_1, embedding_2))
# ## tensor([[0.6003]])