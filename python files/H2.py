import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

#LOAD DATASETS
def load_data():
    train_f = pd.read_excel('features/H_dev_11_02_180_4_60_trainFM.xls', header=None)
    develop_f = pd.read_excel('features/H_dev_11_02_180_4_60_testFM.xls', header=None)
    train_labels = pd.read_excel('labels/train_label.xls', header=None)
    develop_labels = pd.read_excel('labels/development_label.xls', header=None)
    return train_f, develop_f, train_labels, develop_labels

#PREPROCESS DATA
def preprocess_data(features, labels=None, scaler=None):
    """
    Apply z-score standardization to features.
    
    Args:
        features: pandas DataFrame of input features
        labels: pandas DataFrame of target labels (optional)
        scaler: pre-fitted StandardScaler (optional)
    Returns:
        X: standardized feature matrix (numpy array)
        y: target labels array (or None)
        scaler: fitted StandardScaler object for reuse
    """
    X = features.values
    if labels is not None:
        y = labels.values.flatten()
    else:
        y = None
    
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler

#TRAINING AND MODEL EVALUATION
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    #ACCURACY AND METRICS
    accuracy = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
    print("Cohen's Kappa: ", kappa)
    print(classification_report(y_val, y_pred, zero_division=0))
    
    #CONFUSION MATRIX
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()
    
    #ROC CURVE AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {model.__class__.__name__}')
        plt.legend(loc='lower right')
        plt.show()

#MAIN EXECUTION
if __name__ == "__main__":
    train_f, develop_f, train_labels, develop_labels = load_data()
    
    X_train, y_train, scaler = preprocess_data(train_f, train_labels)
    X_val, y_val, _ = preprocess_data(develop_f, develop_labels, scaler=scaler)

    #SMOTE - handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #MODELS TO EVALUATE
    models = {
        'SVC': SVC(kernel='linear', class_weight='balanced', probability=True),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=0)
    }

    #EVALUATE EACH MODEL
    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        train_and_evaluate(model, X_train_resampled, y_train_resampled, X_val, y_val)

    #XGBoost Model
    xgb_model = XGBClassifier(eval_metric='logloss')
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    print("\nBest parameters for XGBoost: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    #Validate best XGBoost model
    best_model = grid_search.best_estimator_
    print("\nEvaluating best XGBoost model on development set:")
    train_and_evaluate(best_model, X_train_resampled, y_train_resampled, X_val, y_val)