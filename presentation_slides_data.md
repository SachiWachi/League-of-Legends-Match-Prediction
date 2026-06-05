# League of Legends Match Outcome Prediction - Presentation Data

## Slide 1: Introduction
- **Goal:** Predict the winner of a League of Legends match.
- **Dataset:** Over 50,000 ranked EUW games.
- **Problem:** Full match data contains leakage (e.g., total towers destroyed directly correlates with winning). We must separate Early-Game predictability from Full-Match hindsight.

## Slide 2: Early-Game Model (The 'Predictive' Model)
- **Features Used:** Champion Draft (Team Composition), Bans, First Blood, First Tower, First Dragon, First Rift Herald.
- **Accuracy:** 72.82%
- **ROC-AUC:** 0.80
- **Top Predictive Features:** firstTower, firstDragon, firstRiftHerald, firstBlood, champ_40
- **Insight:** We can predict the winner with reasonable accuracy before the 15-minute mark just by looking at the draft and the very first objectives.

## Slide 3: Full-Match Model (The 'Hindsight' Model)
- **Features Used:** Early-Game features + Late-Game Objectives (First Baron, First Inhibitor) + Total Objective Differences.
- **Accuracy:** 96.68%
- **ROC-AUC:** 0.99
- **Top Predictive Features:** tower_diff, inhibitor_diff, firstInhibitor, dragon_diff, baron_diff
- **Insight:** Unsurprisingly, models using end-of-game statistics achieve near-perfect accuracy, demonstrating obvious data leakage if used for true forecasting.

## Slide 4: Comeback Analysis & Match Dynamics
- **What is a Comeback?** Matches where the Early-Game model confidently predicts Team A will win (based on draft and early leads), but Team B actually wins the match.
- **Statistics:** Out of 10211 test games, 2594 (25.4%) were successful comeback victories.
- **Conclusion:** League of Legends is not entirely decided in the draft or first 10 minutes. Late-game decision making, team fighting, and objective steals (Baron/Elder) create a measurable threshold of unpredictability.

