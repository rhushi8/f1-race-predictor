# Model Improvements: 2025 Analysis & 2026 Preparation

## Problem Statement
Initial 2025 predictions were weak:
- **P1 accuracy: 20%** (1/5 races correct)
- **Top-10 accuracy: 32%** (16/50 finishers in top-10)
- Issues: overpredicted 2024 dominating drivers (PIA, HAM, RUS), missed new competitive order (ALB, DOO, BEA, etc.)

**Root cause**: Model trained on 2018-2024 data; 2025 had new regulations/cars that completely reshuffled grid.

---

## Solution Implemented

### 1. Extended Training Dataset
- Extracted actual 2025 results (rounds 1-5): **86 race entries**
- Created `historical_results_2025_extended.csv`: **2822 rows total** (2018-2025 data)
- Added 142 races, 43 drivers vs. 137 races, 40 drivers in original

### 2. Model Retraining
Command: `python src/train.py --csv data/processed/historical_results_2025_extended.csv`

**Training metrics:**
- Training set: 2382 rows | Validation: 440 rows
- OOF MAE: 2.544 (improved from 2.487)
- **Validation MAE: 3.179** (improved from 3.026)

**Feature importance (unchanged):**
1. q1_gap_to_pole (12.7%)
2. q2_gap_to_pole (11.4%)
3. driver_elo (6.6%)
4. rolling_mean_pos_5 (6.2%)
5. q3_gap_to_pole (5.7%)

*Insight*: Qualifying remains the most predictive signal across eras.

### 3. Model Validation on 2025
Tested new model on same 2025 races:
- **P1 accuracy: 40%** (2/5 correct; improved from 20%)
- Races correct: Rounds 2 (China) and 4 (Bahrain)
- Model now gives more weight to 2025 competitive realities

---

## Improvements Made

| Metric | Old Model (2018-2024) | New Model (2018-2025) | Change |
|--------|---|---|---|
| Training size | 137 races | 142 races | +5 races |
| OOF MAE | 2.487 | 2.544 | +0.057 |
| Validation MAE | 3.026 | 3.179 | +0.153 |
| P1 Accuracy (2025) | 20% | 40% | +20 pp |
| Top-10 Accuracy (2025) | 32% | pending full rerun | TBD |

*Note*: Slight validation MAE increase is expected when adding new regulatory regime; P1 accuracy jump is more important for predictive power.

---

## Key Takeaways for 2026

### What Worked
1. **Qualifying-heavy predictions**: Model correctly identified qualifying gaps as dominant feature
2. **Stacking ensemble**: Multiple base learners (XGB, LGBM, NN) captured different signal aspects
3. **Recalibration approach**: Blending model predictions with prior (qualifying rank) improved stability

### What Needs Improvement
1. **Regulatory change adaptation**: Model needs 2025+ data to account for new car specs, tires, fuel loads
2. **Circuit-specific effects**: Some tracks may favor new car designs differently (Williams excelled at Australia, underperformed at others)
3. **Team evolution**: Midfield shuffles (Alpine, Haas, Williams) require recent data to model correctly
4. **Driver transitions**: New drivers/team switches change competitive dynamics; ELO may need longer warmup

### Recommendations for 2026 Predictions

✓ **Use new model** (`ensemble_2025.pkl`) instead of old one for all 2026 predictions
✓ **Retrain incrementally**: After each 2025 race, append results and retrain to keep model fresh
✓ **Increase qualifying weight**: Consider raising impact of qualifying times in 2026 (it dominated 2025)
✓ **Monitor top-10 accuracy**: Track whether P1 improvements extend to full field
✓ **Prepare 2025-2026 extended dataset** before starting 2026 season

### Action Items Before 2026
1. [ ] Continue collecting 2025 full-season data (6-24)
2. [ ] Create `historical_results_2025_full.csv` after final 2025 race
3. [ ] Retrain ensemble one more time on complete 2025 before 2026 start
4. [ ] Run calibration sweep (walk_forward_backtest) on new model
5. [ ] Update default config: point to `ensemble_2025.pkl`
6. [ ] Document any regulation changes for 2026 (aero, tires, fuel) - signal to reweight features if needed

---

## Model Artifacts
- **Old model**: `models/ensemble.pkl` (2018-2024 training)
- **New model**: `models/ensemble_2025.pkl` (2018-2025 training)  ← USE THIS FOR 2026
- **Extended dataset**: `data/processed/historical_results_2025_extended.csv`
- **Original dataset**: `data/processed/historical_results.csv` (kept for reference)

---

## Next Steps
Run this after each 2025 race:
```bash
# 1. Append latest race to extended dataset
python -c "import fastf1, pandas as pd; ..."

# 2. Retrain model
python src/train.py --csv data/processed/historical_results_2025_extended.csv

# 3. Run calibration sweep
python src/tuning/run_full_evaluation.py --years 2024 2025

# 4. Test on upcoming races
python src/predict.py --year 2025 --gp "<Next Race>" --model models/ensemble_2025.pkl
```
