from predict import TextPredictor

def run_tests():
    predictor = TextPredictor()

    ejemplos = [
        "How can I install Python on Ubuntu?",
        "This question is not related to programming at all",
        "Why my code does not compile?"
    ]

    print("=== Pruebas de prediccion ===")
    for texto in ejemplos:
        pred = predictor.predict(texto)
        print(f"Texto: {texto}")
        print(f"Prediccion: {pred}")

if __name__ == "__main__":
    run_tests()