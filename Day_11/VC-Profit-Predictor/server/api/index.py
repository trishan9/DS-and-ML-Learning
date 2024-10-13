from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import tempfile
import os

from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "model-file" in request.files:
        model_file = request.files["model-file"]
        if model_file.filename.endswith(".pkl"):
            model = joblib.load(model_file)
        else:
            return jsonify({"error": "Unsupported model file format. Use .pkl."}), 400
    else:
        model = joblib.load("./model/linear_regression_model.pkl")

    # Parse input data
    data = request.form
    RnD_Spend = float(data["RnD_Spend"])
    Administration = float(data["Administration"])
    Marketing_Spend = float(data["Marketing_Spend"])
    State = data["State"]

    # One-hot encode the 'State' field
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


@app.route("/api/train", methods=["POST"])
def train_model():
    model_filename = None
    try:
        model_name = request.form.get("modelName")
        file = request.files["file"]

        if not model_name or not file:
            return jsonify({"error": "Model name or file is missing"}), 400

        if file.filename.endswith(".csv"):
            data = pd.read_csv(file)
        elif file.filename.endswith(".json"):
            data = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or JSON."}), 400

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [-1])], remainder="passthrough"
        )
        X = np.array(ct.fit_transform(X))

        model = LinearRegression()
        model.fit(X, y)

        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pkl"
        ) as temp_model_file:
            model_filename = temp_model_file.name
            print("Joblib Start")
            joblib.dump(model, model_filename)
            print("Joblib Done")

        # Send the file to the client for download
        return send_file(model_filename, as_attachment=True)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while training the model."}), 500
    finally:
        if model_filename and os.path.exists(model_filename):
            os.remove(model_filename)
            print(f"Deleted model file: {model_filename}")


if __name__ == "__main__":
    app.run()
