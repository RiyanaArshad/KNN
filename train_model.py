import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("student_performance.csv")

# Rename columns for consistency
df.rename(columns={
    'Study_Hours': 'study_hours',
    'Attendance_Percentage': 'attendance',
    'Previous_Test_Score': 'test_score',
    'Extracurricular': 'extracurricular',
    'Internet_Access': 'internet_access',
    'Parents_Education': 'parents_education',
    'Final_Result': 'performance'
}, inplace=True)

# Convert categorical values to numerical
df['extracurricular'] = df['extracurricular'].map({"Yes": 1, "No": 0})
df['internet_access'] = df['internet_access'].map({"Yes": 1, "No": 0})
df['parents_education'] = df['parents_education'].map({"High School": 0, "College": 1, "Graduate": 2})

# Convert performance to numerical values (Target variable)
df['performance'] = df['performance'].map({"Pass": 1, "Fail": 0})

# Define features and target variable
X = df[['study_hours', 'attendance', 'test_score', 'extracurricular', 'internet_access', 'parents_education']]
y = df['performance']

# Check for missing values
if X.isnull().sum().sum() > 0:
    print("⚠️ Warning: Missing values detected in dataset!")
    X.fillna(0, inplace=True)  # Fill missing values with 0

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model saved as model.pkl and scaler.pkl!")
