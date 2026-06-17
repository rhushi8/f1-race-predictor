# F1 Race Predictor 🏎️

Predict tomorrow's Formula 1 race result using qualifying pace, practice telemetry,
tire degradation analysis, and 10,000 Monte Carlo simulations.

## Requirements

- Python 3.10+
- pip 23+
- Windows/Linux/macOS

Optional but recommended:
- PyTorch (if unavailable, TireDegNN falls back to LightGBM automatically)

## Architecture

```
Data Ingestion
  ├── FastF1 API       → qualifying times, practice laps, telemetry, weather
  └── OpenF1 API       → stints, pit stops, live timing, car data

Feature Engineering
  ├── ELO ratings      → driver & team historical strength
  ├── Circuit affinity → driver's historical record at this track
  ├── Tire deg slope   → seconds lost per lap on each compound
  ├── Pace features    → best lap, long-run pace, fuel-corrected pace
  └── Interaction feats → pace × deg, quali vs practice delta

ML Ensemble (stacking)
  ├── XGBoost          → finish position regression
  ├── LightGBM         → race pace regression
  ├── Neural net       → tire degradation regression (PyTorch)
  └── Logistic reg.    → DNF / safety car probability
  └── Ridge meta-learner  → combines OOF predictions

Monte Carlo (10,000 simulations)
  ├── Lap-time noise sampling
  ├── Tire degradation per stint
  ├── Pit-stop timing variance
  ├── Safety car deployment
  ├── DNF sampling
  └── Weather perturbation

Outputs
  ├── Win / podium / points probabilities
  ├── Expected finish position + 90% CI
  ├── DNF probability per driver
  └── Optimal strategy recommendation
```

## Project structure

```
f1_predictor/
├── config/
│   └── settings.py          # constants, hyper-params, feature lists
├── src/
│   ├── ingestion/
│   │   ├── fastf1_loader.py  # FastF1 session loading + feature extraction
│   │   └── openf1_client.py  # OpenF1 REST API client
│   ├── features/
│   │   └── engineer.py       # ELO, affinity, interactions, imputation, scaling
│   ├── models/
│   │   └── ensemble.py       # XGB + LGBM + NN + Logit stacking ensemble
│   ├── simulation/
│   │   └── monte_carlo.py    # race simulation engine + strategy optimizer
│   ├── tuning/
│   │   ├── walk_forward_backtest.py # strict out-of-sample evaluator + calibration sweep
│   │   ├── calibration_report.py    # multi-season calibration aggregation
│   │   └── run_full_evaluation.py   # one-command multi-year evaluation + report
│   ├── predict.py            # end-to-end inference pipeline (CLI)
│   └── train.py              # model training script (CLI)
├── data/
│   ├── raw/                  # downloaded CSVs
│   ├── processed/            # feature matrices + predictions
│   └── cache/fastf1/         # FastF1 disk cache
├── models/                   # saved model artifacts (.pkl)
├── notebooks/                # Jupyter exploration
└── requirements.txt
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify installation
python -c "import fastf1, pandas, sklearn; print('Dependencies OK')"

# 4. Predict tomorrow's race (heuristic mode, no training needed)
python src/predict.py --year 2024 --gp Bahrain --sims 10000

# 5. Train on historical data
python src/train.py --csv data/processed/historical_results.csv

# 6. Predict with trained model
python src/predict.py --year 2024 --gp Bahrain --sims 10000 \
       --model models/ensemble.pkl

# 7. Strict walk-forward validation + calibration sweep
python src/tuning/walk_forward_backtest.py --year 2024 --optimize-calibration --lock-best

# 8. One-command multi-year evaluation (reuses existing sweeps)
python src/tuning/run_full_evaluation.py --years 2020 2021 2022 2023 2024

# 9. Force full re-run for all listed years
python src/tuning/run_full_evaluation.py --years 2020 2021 2022 2023 2024 --force
```

## Building a historical dataset

The training pipeline expects a CSV with these columns:

| Column | Description |
|---|---|
| `year` | Season year |
| `gp` | Grand Prix name |
| `circuit_id` | Short circuit identifier |
| `driver_code` | 3-letter driver abbreviation |
| `team_name` | Constructor name |
| `grid_position` | Starting grid position |
| `finish_position` | Official finishing position |
| `dnf` | 1 if did not finish, 0 otherwise |
| `q1_time_s` | Q1 best lap (seconds) |
| `q2_time_s` | Q2 best lap (seconds) |
| `q3_time_s` | Q3 best lap (seconds) |
| `fp2_best_lap_s` | FP2 single-lap best |
| `fp2_long_run_pace_s` | FP2 median long-run lap time |
| `deg_slope_medium` | Tire degradation slope on mediums |
| `air_temp_c` | Air temperature |
| `track_temp_c` | Track temperature |

You can build this automatically by running `build_weekend_features()` for
multiple seasons and concatenating the results with the official race finishing
order from FastF1.

## Extending the pipeline

### Add a new feature
1. Compute it in `src/features/engineer.py`
2. Add the column name to `ALL_FEATURES` in `config/settings.py`
3. Add an imputation default in `IMPUTATION_DEFAULTS`

### Add a new base model
1. Create a class with `.fit(X, y)` and `.predict(X)` in `src/models/ensemble.py`
2. Add an instance to `F1StackingEnsemble.base_models`

### Add a new circuit profile
Add an entry in `config/circuits.json`:

```json
"interlagos": {
  "name": "Autodromo Jose Carlos Pace",
  "total_laps": 71,
  "safety_car_rate": 0.10,
  "overtaking_factor": 1.3,
  "weather_rain_prob": 0.35
}
```

### Tune hyperparameters with Optuna
```python
import optuna
from src.models.ensemble import PositionXGB
import xgboost as xgb

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    # ... cross-validate and return MAE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```
