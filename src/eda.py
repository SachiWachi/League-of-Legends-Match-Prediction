import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def configure_aesthetics():
    """Configure modern plot aesthetics."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.dpi'] = 150

def run_eda(df, out_dir="plots"):
    """
    Run basic Exploratory Data Analysis and save plots.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    configure_aesthetics()

    print("Generating EDA plots...")
    
    # 1. Win Distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='winner', palette=['#3498db', '#e74c3c'])
    plt.title("Match Outcome Distribution", weight='bold')
    plt.xlabel("Winning Team (1=Blue, 2=Red)")
    plt.ylabel("Number of Matches")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "win_distribution.png"))
    plt.close()

    # 2. Objective correlation with Winner
    objectives = ['firstBlood', 'firstTower', 'firstDragon', 'firstRiftHerald', 'firstBaron', 'firstInhibitor']
    
    # We want to see how often getting the first objective leads to a win
    # Create a summary dataframe
    win_rates = []
    for obj in objectives:
        for team in [1, 2]:
            # Filter matches where 'team' got the first objective
            got_obj = df[df[obj] == team]
            if len(got_obj) > 0:
                wins = len(got_obj[got_obj['winner'] == team])
                win_rate = wins / len(got_obj)
                win_rates.append({'Objective': obj.replace('first', 'First '), 'Team': f"Team {team}", 'Win Rate': win_rate})

    win_rates_df = pd.DataFrame(win_rates)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=win_rates_df, x='Objective', y='Win Rate', hue='Team', palette=['#3498db', '#e74c3c'])
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    plt.title("Win Rate when Securing First Objectives", weight='bold')
    plt.ylabel("Win Rate")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "objective_win_rates.png"))
    plt.close()

    # 3. Game Duration Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gameDuration'] / 60, bins=50, kde=True, color="#9b59b6")
    plt.title("Game Duration Distribution", weight='bold')
    plt.xlabel("Game Duration (Minutes)")
    plt.ylabel("Frequency")
    plt.axvline((df['gameDuration'] / 60).mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Duration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "game_duration.png"))
    plt.close()

    print(f"EDA plots saved to {out_dir}/")

if __name__ == "__main__":
    from data_loader import load_data
    df, _ = load_data("..")
    run_eda(df, "../plots")
