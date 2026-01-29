import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import json
import random

from tensorflow.keras.models import load_model

# -------------------------
# Load model & data
# -------------------------
model = load_model(
    '/Users/amit/Desktop/PythonProjects/Chatbot/chatbot/chatbot_model.h5'
)

intents = json.loads(
    open('/Users/amit/Desktop/PythonProjects/Chatbot/intents.json').read()
)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# -------------------------
# NLP helpers
# -------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower()) for word in sentence_words
    ]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [
        {"intent": classes[i], "probability": float(r)}
        for i, r in enumerate(res)
        if r > ERROR_THRESHOLD
    ]

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def getResponse(ints, intents_json):
    if not ints:
        return "Sorry, I didn’t understand that. Can you rephrase?"

    tag = ints[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn’t understand that."


def chatbot_response(msg):
    ints = predict_class(msg, model)
    return getResponse(ints, intents)

# -------------------------
# Tkinter GUI
# -------------------------
import tkinter as tk
from tkinter import *

def send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg:
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(False, False)

# Chat window
ChatLog = Text(
    base,
    bd=0,
    bg="white",
    fg="black",
    height=8,
    width=50,
    font=("Verdana", 12)
)
ChatLog.config(state=DISABLED)

# Scrollbar
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog["yscrollcommand"] = scrollbar.set

# Entry box
EntryBox = Text(
    base,
    bd=0,
    bg="white",
    fg="black",
    insertbackground="black",
    width=29,
    height=5,
    font=("Verdana", 12)
)

# Send button
SendButton = Button(
    base,
    font=("Verdana", 12, "bold"),
    text="Send",
    width=10,
    height=5,
    bd=0,
    bg="#131515",
    fg="#e21616",
    command=send
)

# Layout
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
