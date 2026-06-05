import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data
from src.eda import run_eda
from src.features import prepare_early_game_features, prepare_full_match_features, get_train_test_splits
from src.models import train_and_compare_models, save_model
from src.evaluate import evaluate_and_document, generate_presentation_docs
import pandas as pd

def main():
    print("="*50)
    print("League of Legends Match Outcome Prediction Pipeline")
    print("="*50)

    # 1. Load Data
    df, champ_map = load_data()

    # 2. Exploratory Data Analysis
    print("\n[1/4] Running Exploratory Data Analysis...")
    run_eda(df, out_dir="plots")

    # 3. Feature Engineering & Modeling (Early Game)
    print("\n[2/4] Training Early-Game Model...")
    X_early, y_early = prepare_early_game_features(df)
    X_train_e, X_test_e, y_train_e, y_test_e, scaler_e = get_train_test_splits(X_early, y_early)
    
    best_early_model, early_model_name, early_results = train_and_compare_models(
        X_train_e, y_train_e, X_test_e, y_test_e, prefix="Early-Game"
    )
    save_model(best_early_model, scaler_e, "early_model.joblib")

    # 4. Feature Engineering & Modeling (Full Match)
    print("\n[3/4] Training Full-Match Model...")
    X_full, y_full = prepare_full_match_features(df)
    X_train_f, X_test_f, y_train_f, y_test_f, scaler_f = get_train_test_splits(X_full, y_full)
    
    best_full_model, full_model_name, full_results = train_and_compare_models(
        X_train_f, y_train_f, X_test_f, y_test_f, prefix="Full-Match"
    )
    save_model(best_full_model, scaler_f, "full_model.joblib")

    # 5. Evaluation and Documentation
    print("\n[4/4] Evaluating Models and Generating Documentation...")
    early_summary, y_pred_early = evaluate_and_document(best_early_model, X_test_e, y_test_e, "Early-Game")
    full_summary, y_pred_full = evaluate_and_document(best_full_model, X_test_f, y_test_f, "Full-Match")
    
    # Comeback analysis: Where Early predicted one way, but Full/Ground Truth went the other way.
    # Predict Team 1 wins early (y_pred_early == 1) but actually Team 2 wins (y_test_e == 0)
    # Or Predict Team 2 wins early (y_pred_early == 0) but actually Team 1 wins (y_test_e == 1)
    # A true comeback is when early prediction is WRONG but the game outcome is what it is.
    early_wrong = (y_pred_early != y_test_e)
    # To filter interesting comebacks, maybe the full match model got it right (predictable late game)
    full_right = (y_pred_full == y_test_f)
    
    comeback_cases = (early_wrong & full_right).sum()
    total_test = len(y_test_e)
    
    generate_presentation_docs(early_summary, full_summary, comeback_cases, total_test)

    print("\nPipeline Complete!")
    print("Next steps:")
    print("1. Review plots in the `plots/` directory.")
    print("2. Review `presentation_slides_data.md` for your presentation.")
    print("3. Run `streamlit run app.py` to launch the interactive UI.")

if __name__ == "__main__":
    main()
