
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Add this import

app = Flask(__name__)
CORS(app) 

# Load the trained linear regression model
model_filename_pkl = 'linear_regression_model.pkl'
model = joblib.load(model_filename_pkl)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    tv = float(data['tv'])
    radio = float(data['radio'])
    social_media = float(data['socialMedia'])
    
    influencer_mapping = {'Macro': 3, 'Mega': 4, 'Micro': 1, 'Nano': 2}
    influencer_value = influencer_mapping.get(data['influencerType'], 0)

    input_data = np.array([[tv, radio, social_media, influencer_value]])
    prediction = model.predict(input_data)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
