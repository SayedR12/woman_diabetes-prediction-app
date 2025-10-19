import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model safely
MODEL_PATH = "women diabetes model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(" Model loaded successfully.")
except Exception as e:
    model = None
    print(" Failed to load model:", e)

@app.route("/")
def home():
    return "woman Diabetes Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    features = data.get("features", [])
    
    if not features:
        return jsonify({"error": "No features provided"}), 400
    
    try:
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

