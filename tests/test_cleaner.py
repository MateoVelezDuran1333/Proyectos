from src.cleaners import CleanerDataProcessor

cleaner = CleanerDataProcessor()

#Datos de prueba 

texts = [
    'Hello!!! This is a TEST 😊😊 <b>HTML</b> with https://example.com',
    'What are the best ways to LEARN Python programming?',
    'Im trying to CLEAN [duplicate] text & remove STOPWORDS.'
]

#Transformar

processed = cleaner.transform(texts)

#Resultados

print("=== Texto original ===")
for t in texts:
    print(t)

print("/n === Texto procesado ===")
for p in processed:
    print(p)