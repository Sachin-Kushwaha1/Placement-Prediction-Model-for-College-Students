from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('placement_model.pkl')

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')  # Ensure the scaler was saved during training

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form data from the request
            cgpa = float(request.form['cgpa'])
            internships = int(request.form['internships'])
            projects = int(request.form['projects'])
            certifications = int(request.form['certifications'])
            extracurricular = int(request.form['extracurricular'])
            aptitude = int(request.form['aptitude'])
            mock = int(request.form['mock'])

            # Combine all features into a numpy array for prediction
            input_data = np.array([[cgpa, internships, projects, certifications, extracurricular, aptitude, mock]])
            
            # Apply scaling to the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the loaded model
            prediction = model.predict(input_data_scaled)

            # Interpret the result
            output = 'Placed' if prediction[0] == 1 else 'Not Placed'

            # Render the result on the HTML page
            return render_template('index.html', prediction_text=f'The student will be {output}')
        
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
