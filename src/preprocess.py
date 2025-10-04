import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessData(BaseEstimator, TransformerMixin):

    def __init__(self, text_columns = None, label_column = None, sep = ','):
        """
        Parámetros:
        -----------
        text_columns : list[str]
            Columnas de texto a concatenar.
        label_column : str
            Columna objetivo.
        sep : str
            Separador de CSV.
        """

        self.text_columns = text_columns
        self.label_column = label_column
        self.sep = sep

        #Diccionario de clases

        self.classes_mapping = {
            'open' : 1,
            'too localized' : 2,
            'not a real question' : 3,
            'off topic' : 4,
            'not constructive' : 5
        }

    def fit(self, X, y=None):
        return self #No se ajusta a nada, solo se transforma
    
    def transform(self, X, y=None):

        """
        X puede ser:
        - Un Dataframe ya cargado
        - Una ruta de CSV
        """
        if isinstance(X, str): #Si X es la ruta al csv
            df = pd.read_csv(X, sep=self.sep)
        else: #si ya es un dataframe
            df = X.copy()

        #Concatenar las columnas de texto
        df['text_complete'] = df[self.text_columns].astype(str).agg(" ".join, axis=1)

        #Mapear etiquetas
        if self.label_column and (self.label_column in df.columns): #Primero verifica que no sea None el label_column y luego verifica si esta en la columna
            df[self.label_column] = df[self.label_column].map(self.classes_mapping)
            return df['text_complete'].values, df[self.label_column].values
        return df['text_complete'].values, None