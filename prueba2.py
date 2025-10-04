import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Importar tus módulos
from src.preprocess import PreprocessData
from src.cleaners import CleanerDataProcessor

# ==============================
# 1. Configuración
# ==============================
DATA_PATH = "data/dataset.csv"
TEXT_COLUMNS = ["Title", "BodyMarkdown"]   # <-- Ajusta según tu dataset
LABEL_COLUMN = "OpenStatus"
MODEL_PATH = "models/text_model.pkl"

# ==============================
# 2. Preprocesamiento
# ==============================
preprocessor = PreprocessData(
    text_columns=TEXT_COLUMNS,
    label_column=LABEL_COLUMN
)

# Cargar y transformar dataset
X, y = preprocessor.transform(DATA_PATH)

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3. Pipeline de NLP + Clasificación
# ==============================
pipeline = Pipeline([
    ("cleaner", CleanerDataProcessor()),       # Limpieza + lematización
    ("tfidf", TfidfVectorizer()),              # Vectorización TF-IDF
    ("clf", SGDClassifier(random_state=42))    # Clasificador lineal
])

# ==============================
# 4. Búsqueda de Hiperparámetros
# ==============================
param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__loss": ["hinge", "log_loss"],  # SVM lineal o Regresión Logística
    "clf__alpha": [1e-4, 1e-3, 1e-2]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# ==============================
# 5. Evaluación
# ==============================
print("\nMejores hiperparámetros encontrados:")
print(grid.best_params_)

y_pred = grid.predict(X_test)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# 6. Guardar modelo entrenado
# ==============================
joblib.dump(grid.best_estimator_, MODEL_PATH)
print(f"\nModelo guardado en {MODEL_PATH}")
