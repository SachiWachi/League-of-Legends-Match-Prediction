import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_and_document(model, X_test, y_test, prefix, out_dir="plots"):
    """Evaluates the model, saves confusion matrix, and returns a summary dict."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = np.mean(y_pred == y_test)
    roc = roc_auc_score(y_test, y_prob)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Team 2 Win', 'Team 1 Win'], yticklabels=['Team 2 Win', 'Team 1 Win'])
    plt.title(f"{prefix} Model Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_confusion_matrix.png"))
    plt.close()
    
    # Feature Importances
    importances = None
    if hasattr(model, "coef_"):
        importances = pd.Series(model.coef_[0], index=X_test.columns)
    elif hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        
    if importances is not None:
        # Plot top 10 features
        plt.figure(figsize=(10, 6))
        importances.abs().sort_values(ascending=False).head(10).plot(kind='barh', color='#2ecc71')
        plt.title(f"Top 10 Feature Importances ({prefix})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_feature_importance.png"))
        plt.close()

    summary = {
        "accuracy": acc,
        "roc_auc": roc,
        "top_features": importances.abs().sort_values(ascending=False).head(5).index.tolist() if importances is not None else []
    }
    return summary, y_pred

def generate_presentation_docs(early_summary, full_summary, comeback_count, total_test):
    """Generates the presentation_slides_data.md file."""
    content = f"""# League of Legends Match Outcome Prediction - Presentation Data

## Slide 1: Introduction
- **Goal:** Predict the winner of a League of Legends match.
- **Dataset:** Over 50,000 ranked EUW games.
- **Problem:** Full match data contains leakage (e.g., total towers destroyed directly correlates with winning). We must separate Early-Game predictability from Full-Match hindsight.

## Slide 2: Early-Game Model (The 'Predictive' Model)
- **Features Used:** Champion Draft (Team Composition), Bans, First Blood, First Tower, First Dragon, First Rift Herald.
- **Accuracy:** {early_summary['accuracy']:.2%}
- **ROC-AUC:** {early_summary['roc_auc']:.2f}
- **Top Predictive Features:** {', '.join(early_summary['top_features'])}
- **Insight:** We can predict the winner with reasonable accuracy before the 15-minute mark just by looking at the draft and the very first objectives.

## Slide 3: Full-Match Model (The 'Hindsight' Model)
- **Features Used:** Early-Game features + Late-Game Objectives (First Baron, First Inhibitor) + Total Objective Differences.
- **Accuracy:** {full_summary['accuracy']:.2%}
- **ROC-AUC:** {full_summary['roc_auc']:.2f}
- **Top Predictive Features:** {', '.join(full_summary['top_features'])}
- **Insight:** Unsurprisingly, models using end-of-game statistics achieve near-perfect accuracy, demonstrating obvious data leakage if used for true forecasting.

## Slide 4: Comeback Analysis & Match Dynamics
- **What is a Comeback?** Matches where the Early-Game model confidently predicts Team A will win (based on draft and early leads), but Team B actually wins the match.
- **Statistics:** Out of {total_test} test games, {comeback_count} ({comeback_count/total_test:.1%}) were successful comeback victories.
- **Conclusion:** League of Legends is not entirely decided in the draft or first 10 minutes. Late-game decision making, team fighting, and objective steals (Baron/Elder) create a measurable threshold of unpredictability.

"""
    with open("presentation_slides_data.md", "w", encoding='utf-8') as f:
        f.write(content)
    print("Generated presentation_slides_data.md")
