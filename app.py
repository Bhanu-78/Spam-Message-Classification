from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        msg = request.form["message"]

        if msg.strip():
            sample = vectorizer.transform([msg])
            prediction = model.predict(sample)[0]
            confidence = max(model.predict_proba(sample)[0]) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)