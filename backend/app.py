# app.py - API Flask du chatbot étudiant

import json
import numpy as np
import nltk
import pickle
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# ── Charger le modèle et les données ──
model   = load_model("model/chatbot_model.h5")
words   = pickle.load(open("model/words.pkl",   "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

print("✅ Modèle chargé avec succès !")

# ── Fonctions du chatbot ──

def clean_sentence(sentence):
    # Découper et lemmatiser la phrase
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    # Convertir la phrase en vecteur 0/1
    tokens = clean_sentence(sentence)
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    # Prédire le tag de la phrase
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]), verbose=0)[0]

    # Ignorer les prédictions trop faibles
    ERROR_THRESHOLD = 0.25
    results = [
        {"intent": classes[i], "probability": float(r)}
        for i, r in enumerate(result)
        if r > ERROR_THRESHOLD
    ]

    # Trier par probabilité
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_response(predicted):
    # Trouver la réponse correspondant au tag prédit
    if not predicted:
        return "Je n'ai pas compris. Peux-tu reformuler ta question ?"

    tag = predicted[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Je n'ai pas de réponse pour ça pour le moment."

# ── Routes ──

@app.route("/")
def home():
    return "✅ Chatbot API fonctionne !"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    # Vérifier que le message existe
    if not data or "message" not in data:
        return jsonify({"error": "Message manquant"}), 400

    message = data["message"]

    # Prédire et répondre
    predicted = predict_class(message)
    response  = get_response(predicted)

    return jsonify({
        "message":  message,
        "response": response,
        "intent":   predicted[0]["intent"] if predicted else "unknown"
    })

# ── Lancer le serveur ──
if __name__ == "__main__":
    app.run(debug=True, port=5000)