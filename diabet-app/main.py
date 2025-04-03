from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Încarcă modelul Keras
model = load_model("best_diabetes_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0][0]
        rezultat = "Pozitiv (risc de diabet)" if prediction > 0.5 else "Negativ (fără risc)"
        return jsonify({"predictie": rezultat})
    except Exception as e:
        return jsonify({"eroare": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
