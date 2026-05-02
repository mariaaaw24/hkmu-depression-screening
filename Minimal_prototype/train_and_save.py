import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

#LOAD DATASET
def load_data():
    train_f = pd.read_excel('features/H_dev_11_02_180_4_60_trainFM.xls', header=None)
    develop_f = pd.read_excel('features/H_dev_11_02_180_4_60_testFM.xls', header=None)
    train_labels = pd.read_excel('labels/train_label.xls', header=None)
    develop_labels = pd.read_excel('labels/development_label.xls', header=None)
    return train_f, develop_f, train_labels, develop_labels

#PREPROCESS DATA
def preprocess_data(features, labels=None):
    X = features.values
    y = labels.values.flatten() if labels is not None else None
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

#TRAINING AND EVALUATION
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
    print("Cohen's Kappa: ", kappa)
    print(classification_report(y_val, y_pred, zero_division=0))

#MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 Fast Training: Head Pose Quaternion + XGBoost")
    
    train_f, develop_f, train_labels, develop_labels = load_data()
    X_train, y_train, scaler = preprocess_data(train_f, train_labels)
    X_val, y_val, _ = preprocess_data(develop_f, develop_labels)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    best_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training optimized XGBoost model...")
    train_and_evaluate(best_model, X_train_resampled, y_train_resampled, X_val, y_val)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    with open('models/best_model_xgboost.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    np.save('data/X_train_resampled.npy', X_train_resampled)
    np.save('data/X_val.npy', X_val)
    
    participant_ids = {
        'train': [f'train_{i}' for i in range(len(X_train_resampled))],
        'val': [f'val_{i}' for i in range(len(X_val))]
    }
    with open('data/participant_ids.pkl', 'wb') as f:
        pickle.dump(participant_ids, f)
    
    print("\n✅FAST TRAINING COMPLETE! Ready for prototype.")