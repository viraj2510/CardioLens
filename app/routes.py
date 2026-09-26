from flask import render_template, request, jsonify, current_app as app
import pickle
import numpy as np
import pandas as pd
import os

CONTACT_EMAIL = "viraj.gundewar1995@gmail.com"
# Development-demo threshold selected to favor sensitivity on the fixed held-out
# split. It is not clinically validated and must not be used for diagnosis.
SCREENING_THRESHOLD = 0.30

# Construct the path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'heart_disease_model.pkl')

# Load the trained model and the column list
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    model_columns = model_data["columns"]
    screening_threshold = model_data.get("threshold", SCREENING_THRESHOLD)
except FileNotFoundError:
    model = None
    model_columns = None
    screening_threshold = SCREENING_THRESHOLD
    print(f"Error: Model file not found at {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    status = 'ok' if model is not None else 'model_unavailable'
    return jsonify({'status': status}), 200 if model is not None else 503

@app.route('/contact')
def contact():
    return render_template('contact.html', contact_email=CONTACT_EMAIL)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        if not isinstance(data, dict) or not data:
            return jsonify({'error': 'Please provide a complete patient profile.'}), 400
        
        # Convert incoming data into a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Convert numeric fields and restore boolean fields from form strings.
        for col in input_df.columns:
            if col in {'fbs', 'exang'}:
                input_df[col] = input_df[col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
            elif col not in {'sex', 'cp', 'restecg', 'slope', 'thal'}:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Apply one-hot encoding
        input_df_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Align the columns of the input data with the columns the model was trained on
        # This adds any missing columns and fills them with 0
        final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Use a sensitivity-oriented research-demo cutoff instead of the
        # classifier's default 0.50. The model score is not clinically calibrated.
        model_score = float(model.predict_proba(final_df)[0, 1])
        raised_flag = model_score >= screening_threshold
        output = (
            'Higher-risk screening flag returned.'
            if raised_flag else 'No higher-risk screening flag returned.'
        )

        return jsonify({
            'prediction_text': output,
            'model_score_percent': round(model_score * 100, 1),
            'screening_threshold_percent': round(screening_threshold * 100, 1),
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400
