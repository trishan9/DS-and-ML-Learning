from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load("./model/linear_regression_model.pkl")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json

    RnD_Spend = float(data["RnD_Spend"])
    Administration = float(data["Administration"])
    Marketing_Spend = float(data["Marketing_Spend"])
    State = data["State"]

    # One-hot encode the 'State' field manually
    state_encoded = [0, 0, 0]
    if State == "california":
        state_encoded = [1, 0, 0]
    elif State == "florida":
        state_encoded = [0, 1, 0]
    elif State == "new-york":
        state_encoded = [0, 0, 1]

    features = state_encoded + [RnD_Spend, Administration, Marketing_Spend]
    prediction = model.predict([features])[0]

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
