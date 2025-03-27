from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Desktop\KNN\student_performance.csv")

# Convert categorical columns to numerical values
label_enc = LabelEncoder()
df["Extracurricular"] = label_enc.fit_transform(df["Extracurricular"])
df["Internet_Access"] = label_enc.fit_transform(df["Internet_Access"])
df["Parents_Education"] = label_enc.fit_transform(df["Parents_Education"])
df["Final_Result"] = label_enc.fit_transform(df["Final_Result"])  # Pass=1, Fail=0

# Prepare data for training
X = df.drop(columns=["Final_Result"])
y = df["Final_Result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

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

            # Predict result
            result = knn.predict(input_data)[0]
            prediction = "Pass 🎓" if result == 1 else "Fail ❌"
        
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
