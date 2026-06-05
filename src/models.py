from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_compare_models(X_train, y_train, X_test, y_test, prefix="model"):
    """
    Trains multiple classification models and returns the best one based on accuracy.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    best_acc = 0
    best_model = None
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} for {prefix}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"--> Best {prefix} model: {best_name} (Acc: {best_acc:.4f})")
    
    return best_model, best_name, results

def save_model(model, scaler, filename="model.joblib"):
    """Saves the trained model and its associated scaler."""
    joblib.dump({'model': model, 'scaler': scaler}, filename)

def load_model(filename="model.joblib"):
    """Loads a trained model and scaler."""
    if os.path.exists(filename):
        return joblib.load(filename)
    return None
