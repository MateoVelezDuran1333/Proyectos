# Usamos una imagen oficial de Python
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y usar stdout/err en lugar de buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias del sistema necesarias para spaCy y NLTK
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo de spaCy y stopwords de NLTK
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader stopwords

# Copiar el resto del proyecto
COPY . .

# Asegurarnos de que Python vea /app como ruta base
ENV PYTHONPATH=/app

# Exponer el puerto de la API
EXPOSE 8000

# Comando para correr la API con Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
