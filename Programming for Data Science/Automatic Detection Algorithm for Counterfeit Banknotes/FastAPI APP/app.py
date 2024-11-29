from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


model = joblib.load("logistic_regression_model.joblib")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
   return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(
   request: Request ,
   diagonal_size: float = Form(...),
   height_left: float = Form(...),
   height_right: float = Form(...),
   lower_margin: float = Form(...),
   upper_margin: float = Form(...),
   length: float = Form(...)
):
   input_data = pd.DataFrame([{
      "diagonal": diagonal_size,
      "height_left": height_left,
      "height_right": height_right,
      "margin_low": lower_margin,
      "margin_up": upper_margin,
      "length": length
   }])

   prediction = model.predict(input_data)
   authenticity = "Genuine" if prediction[0] == 1 else "Counterfeit"

   return templates.TemplateResponse("result.html", {
      "request": request,
      "authenticity": authenticity,
      "input_data": input_data.to_dict(orient="records")[0]
   })