import re
import string
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords



class CleanerDataProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        # Cargar modelo de spaCy (asegúrate de ejecutar antes: python -m spacy download en_core_web_sm)
        #self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat", "tok2vec"])


        # Stopwords en inglés (puedes cambiar a español si corresponde)
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Limpieza básica: minúsculas, quitar URLs, HTML, puntuación, etc."""
        text = str(text).lower()

        # Eliminar texto entre corchetes
        text = re.sub(r"\[.*?\]", "", text)

        # Eliminar URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Eliminar etiquetas HTML
        text = re.sub(r"<.*?>+", "", text)

        # Eliminar puntuación (excepto '?')
        punctuation = string.punctuation.replace("?", "")
        text = re.sub(r"[%s]" % re.escape(punctuation), "", text)

        # Eliminar saltos de línea
        text = re.sub(r"\n", " ", text)

        # Eliminar caracteres no ASCII (emojis, símbolos raros)
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        # Espacios extra
        text = text.strip()

        return text

    def lemmatized_stopwords(self, text: str) -> str:
        """
        Limpieza + lematización con spaCy + eliminación de stopwords.
        """
        clean_data = self.clean_text(str(text))
        doc_en = self.nlp_en(clean_data)

        # Lematizar y quitar stopwords
        lemmatized = [token.lemma_ for token in doc_en if token.text.lower() not in self.stop_words]

        return " ".join(lemmatized).strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.lemmatized_stopwords(str(doc)) for doc in X]

