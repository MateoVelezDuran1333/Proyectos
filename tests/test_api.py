import pytest
from fastapi.testclient import TestClient
from main import app

# Cliente de pruebas de FastAPI
client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Bienvenido a la API de clasificación de texto"

def test_predict_valid():
    response = client.post("/predict", json={"text": "How to install Python on Windows?"})
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], str)

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200  # Modelo debería manejar vacío, pero tú decides si 422 es mejor
    result = response.json()
    assert "prediction" in result

def test_predict_invalid_payload():
    response = client.post("/predict", json={"wrong_field": "hola"})
    assert response.status_code == 422  # Error de validación de Pydantic
