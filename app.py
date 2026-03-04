import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide tensorflow info logs

from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("App file is running...")

app = Flask(__name__)

# -------------------------------
# Load Model & Tokenizer
# -------------------------------

try:
    model = load_model("final_model.keras")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print("Error loading tokenizer:", e)

max_len = 200

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        review = request.form.get("review", "")

        if review.strip() != "":
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=max_len, padding="post")

            pred = model.predict(padded)
            predicted_rating = np.argmax(pred) + 1
            confidence = float(np.max(pred))

            prediction = predicted_rating

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# -------------------------------
# Start Server
# -------------------------------

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
@app.route("/")
def home():
    return "Model removed test working!"