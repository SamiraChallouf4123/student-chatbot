# train.py - Entraîner le modèle NLP du chatbot

import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# ── Charger les intents ──
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

words    = []  # tous les mots
classes  = []  # tous les tags
documents = [] # paires (mots, tag)

ignore_chars = ["?", "!", ".", ","]

# ── Traiter chaque intent ──
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # Découper la phrase en mots
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Associer les mots au tag
        documents.append((word_list, intent["tag"]))

        # Ajouter le tag à la liste
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatiser et nettoyer les mots
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"✅ {len(words)} mots uniques trouvés")
print(f"✅ {len(classes)} catégories : {classes}")

# ── Sauvegarder mots et classes ──
pickle.dump(words,   open("model/words.pkl",   "wb"))
pickle.dump(classes, open("model/classes.pkl", "wb"))

# ── Créer les données d'entraînement ──
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in document[0]]

    # Bag of words : 1 si le mot est présent, 0 sinon
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    # Output : 1 pour le bon tag, 0 pour les autres
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mélanger les données
import random
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print(f"✅ {len(train_x)} exemples d'entraînement créés")

# ── Construire le modèle ──
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax")
])

# Compiler le modèle
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Entraîner le modèle
print("⏳ Entraînement en cours...")
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Sauvegarder le modèle
model.save("model/chatbot_model.h5")
print("✅ Modèle sauvegardé dans model/chatbot_model.h5")