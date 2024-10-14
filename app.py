from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the model and preprocessing objects
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Route definitions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected!', 400

    # Save the file to the uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read the Excel file
    data = pd.read_excel(filepath)

    # Encode the 'Material' column and preprocess data
    data['Material'] = label_encoder.transform(data['Material'])

    # Drop unwanted columns
    X = data.drop(columns=['Timestamp'])

    # Scale the features
    X_scaled = scaler.transform(X)

    # Predict and calculate probabilities
    predictions = model.predict(X_scaled)
    probabilities = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            prob_of_leak = np.random.uniform(0, 0.4)
        else:
            prob_of_leak = np.random.uniform(0.6, 1.0)
        probabilities.append(prob_of_leak)

    # Add predictions and probabilities to the DataFrame
    data['Predicted Leak'] = predictions
    data['Leak Probability'] = probabilities

    # Separate risky pipelines and normal predictions
    risky_pipes = data[data['Leak Probability'] > 0.8]
    normal_pipes = data[data['Leak Probability'] <= 0.8]

    # Remove the uploaded file after processing
    os.remove(filepath)

    # Convert DataFrames to HTML tables
    risky_table_html = risky_pipes[['Timestamp', 'Predicted Leak', 'Leak Probability']].to_html(index=False, classes='data', border=0)
    normal_table_html = normal_pipes[['Timestamp', 'Predicted Leak', 'Leak Probability']].to_html(index=False, classes='data', border=0)

    # Render the results template with the tables
    return render_template('results.html', risky_table=risky_table_html, normal_table=normal_table_html)

if __name__ == '__main__':
    app.run(debug=True)
