from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model and scaler
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    # If model files don't exist, we need to run train_model.py first
    import subprocess
    print("🔄 Training model...")
    subprocess.call(["python", "train_model.py"])
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            test_score = float(request.form["test_score"])
            extracurricular = 1 if request.form["extracurricular"] == "Yes" else 0
            internet_access = 1 if request.form["internet_access"] == "Yes" else 0
            parents_education = {"High School": 0, "College": 1, "Graduate": 2}[request.form["parents_education"]]

            # Prepare input data
            input_data = np.array([[study_hours, attendance, test_score, extracurricular, internet_access, parents_education]])
            
            # Scale the input data using the saved scaler
            input_data_scaled = scaler.transform(input_data)

            # Predict result
            result = model.predict(input_data_scaled)[0]
            prediction = "Pass 🎓" if result == 1 else "Fail ❌"
        
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    # Render will provide PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
