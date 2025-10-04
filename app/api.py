from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import TextPredictor


# Inicializar API
app = FastAPI(title="Text Clasification API")

predictor = TextPredictor()


#Definir el esquema de entrada
class TextInput(BaseModel):
    text : str

@app.get("/")
def home():
    return {"message" : "Bienvenido a la API de clasificación de texto"}

@app.post("/predict")
def predict(input_data : TextInput):
    prediction = predictor.predict([input_data.text])
    return {"prediction" : prediction[0]}