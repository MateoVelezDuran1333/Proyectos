# Clasificación de Preguntas de StackOverflow

Este proyecto implementa un sistema de **clasificación automática de preguntas** de StackOverflow usando **NLP (Procesamiento de Lenguaje Natural)** y **Machine Learning**.  
Se entrenó un modelo con `scikit-learn`, `spaCy` y `nltk`, y se expone mediante una **API con FastAPI**.

---

##  Características
- Preprocesamiento de datos con **NLTK** (stopwords) y **spaCy** (lemmatización).
- Entrenamiento con **SGDClassifier + TF-IDF** dentro de un pipeline.
- Guardado y carga del modelo entrenado con `joblib`.
- API REST para predicción de nuevas preguntas.
- Tests automáticos con `pytest`.

---

## Requisitos

### 1. Clonar repositorio
```bash
git clone https://github.com/usuario/proyecto-modular.git
cd proyecto-modular
```
### 2. Crear entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar recursos adicionales

El proyecto requiere algunos recursos externos:

 NLTK stopwords
 ```bash
import nltk
nltk.download("stopwords")
 ```

Modelo de spaCy
 ```bash
python -m spacy download en_core_web_sm
 ```

### Entrenamiento del modelo

Si deseas reentrenar el modelo desde cero:

python train.py

El modelo entrenado se guarda en:

text_model.pkl


### Probar predicciones
#### 1. Modo script

Puedes probar predicciones rápidas con:
 ```bash
python test.py
 ```

API con FastAPI
Ejecutar servidor
uvicorn api:app --reload


El servidor estará disponible en:

http://127.0.0.1:8000

Endpoints

GET /

{"message": "Bienvenido a la API de clasificación de texto."}


POST /predict
Body:

{
  "text": "How do I install Python on Linux?"
}


Respuesta:

{
  "prediction": "open"
}

### Ejecutar tests
pytest

📂 Estructura del proyecto
proyecto-modular/
│── preprocess.py      # Preprocesamiento de datos
│── cleaners.py        # Limpieza de texto con spaCy y NLTK
│── train.py           # Entrenamiento del modelo
│── predict.py         # Cargador del modelo para predicciones
│── test.py            # Script rápido de pruebas
│── api.py             # API con FastAPI
│── test_api.py        # Tests unitarios de la API
│── requirements.txt   # Dependencias del proyecto
│── text_model.pkl     # Modelo entrenado
