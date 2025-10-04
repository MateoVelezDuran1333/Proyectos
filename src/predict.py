import joblib

class TextPredictor:
    def __init__(self, model_path = './models/text_model.pkl'):
        
        # Cargar el modelo
        self.model = joblib.load(model_path)

        #Diccionario de clases
        self.classes_mapping = {
            1: "open",
            2: "too localized",
            3: "not a real question",
            4: "off topic",
            5: "not constructive"
        }

    def predict(self, texts):
        """
        texts: str o list[str]
        return: list de predicciones (str)
        """
        if isinstance(texts, str):
            texts = [texts] #Convertir a lista
        
        preds = self.model.predict(texts)
        decoded = [self.classes_mapping.get(p, "Unknown") for p in preds]
        return decoded