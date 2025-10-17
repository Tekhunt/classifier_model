from joblib import load

model = load("models/sentiment_model.pkl")
vectorizer = load("models/vectorizer.pkl")

sample = ["I Don't know"]
print("Sample review:", sample)
X_sample = vectorizer.transform(sample)
prediction = model.predict(X_sample)
print("Predicted sentiment:", prediction[0])
