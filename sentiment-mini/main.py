# Mini Sentiment Classifier (muy simple pero funcional)
# Autor: Petrus

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Pequeño dataset de ejemplo (puedes ampliarlo)
train_texts = [
    "me encanta este proyecto, es brutal",
    "esto es increíble y muy útil",
    "qué buena experiencia, repetiría",
    "excelente servicio, súper recomendado",
    "estoy feliz con el resultado",
    "me gusta mucho cómo quedó",
    "no me ha gustado nada",
    "horrible, ha sido un desastre",
    "mala calidad y mala experiencia",
    "estoy muy decepcionado",
    "odio esto, fatal",
    "no lo recomiendo para nada",
]

train_labels = [
    "positivo","positivo","positivo","positivo","positivo","positivo",
    "negativo","negativo","negativo","negativo","negativo","negativo"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
    ("nb", MultinomialNB())
])

model.fit(train_texts, train_labels)

def predict_sentiment(text: str) -> str:
    return model.predict([text])[0]

if __name__ == "__main__":
    print("Mini clasificador de sentimiento 🤖 (escribe 'salir' para terminar)")
    while True:
        s = input("> Tu frase: ").strip()
        if s.lower() in ("salir","exit","quit"):
            break
        if not s:
            continue
        pred = predict_sentiment(s)
        print(f"→ Sentimiento: {pred}")
