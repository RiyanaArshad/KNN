# Student Performance Predictor

A machine learning application that predicts student performance based on various factors including study hours, attendance, and more.

## Features

- Predicts whether a student will pass or fail based on input parameters
- Uses Random Forest Classifier model for accurate predictions
- Responsive web interface

## Deployment on Render

### Prerequisites

- A [Render](https://render.com/) account (free tier is sufficient)
- A GitHub account to store your code

### Steps to Deploy

1. **Push your code to GitHub**
   - Create a new repository on GitHub
   - Initialize your local repository (if not already done)
   ```
   git init
   git add .
   git commit -m "Initial commit"
   ```
   - Link your local repository to GitHub
   ```
   git remote add origin https://github.com/your-username/your-repo-name.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Log in to your Render account
   - Click on "New" and select "Web Service"
   - Connect to your GitHub repository
   - Enter the following settings:
     - **Name**: student-performance-predictor (or any name you prefer)
     - **Environment**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

3. **Verify Deployment**
   - Once the deployment is complete, Render will provide a URL
   - Visit the URL to use your deployed application

## Local Development

To run the application locally:

1. Install the required packages
   ```
   pip install -r requirements.txt
   ```

2. Run the application
   ```
   python app.py
   ```

3. Open a web browser and go to `http://localhost:5000`

## Dataset

The model is trained on the student_performance.csv dataset which includes:
- Study Hours
- Attendance Percentage
- Previous Test Score
- Extracurricular Activities
- Internet Access
- Parents' Education
- Final Result
