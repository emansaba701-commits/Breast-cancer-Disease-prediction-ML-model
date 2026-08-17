
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get input
        input_text = request.form["features"]

        # Convert comma-separated values into numbers
        values = [
            float(x.strip())
            for x in input_text.split(",")
            if x.strip()
        ]

        # Check number of features
        if len(values) != 30:

            return render_template(
                "index.html",
                error=f"Expected 30 features, but received {len(values)} features."
            )

        # Convert to numpy
        input_data = np.array(values).reshape(1, -1)

        # Apply StandardScaler
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            result = "Malignant_cancerous"
        else:
            result = "Benign_Non-cancerous"

        return render_template(
            "index.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
