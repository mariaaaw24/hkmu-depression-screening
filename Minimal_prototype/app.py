from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

#Load model and data
with open('Minimal_prototype/models/best_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Minimal_prototype/data/participant_ids.pkl', 'rb') as f:
    ids = pickle.load(f)

X = np.vstack([
    np.load('Minimal_prototype/data/X_train_resampled.npy'),
    np.load('Minimal_prototype/data/X_val.npy')
])
participants = ids['train'] + ids['val']
features_df = pd.DataFrame(X, index=participants)

@app.route('/')
def index():
    return render_template('index.html', participants=participants)

@app.route('/feature-extraction')
def feature_page():
    return render_template('feature_extraction.html')

@app.route('/predict', methods=['POST'])
def predict():
    pid = request.json.get('participant_id')
    if pid not in features_df.index:
        return jsonify({'error': 'Participant not found'}), 404
    
    feat = features_df.loc[pid].values.reshape(1, -1)
    pred = int(model.predict(feat)[0])
    
    #Confidence from XGBoost probabilities
    if hasattr(model, 'predict_proba'):
        conf = float(model.predict_proba(feat)[0].max())
    else:
        conf = 0.8
    
    return jsonify({
        'screening_id': f'SCR-{pid}',
        'risk_level': 'HIGH' if pred == 1 else 'LOW',
        'confidence': round(conf, 2),
        'recommendation': 'Seek clinical evaluation within 2 weeks' if pred == 1 else 'Low risk detected',
        'note': 'Based on pre-encoded DAIC-WOZ features (no video processed)'
    })

if __name__ == '__main__':
    print("Prototype running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)