# Part 6: Evaluation and Tuning

### 6.1 What this step is
Evaluation quantifies model quality and realism under realistic temporal constraints.
Tuning adjusts parameters for better generalization and calibrated outputs.

### 6.2 Why it is important
Without rigorous evaluation, improvements may be accidental or leakage-driven.
Tuning without proper validation can increase overfitting and reduce race-day reliability.

### 6.3 How it works internally
Core evaluation design:
- holdout by race groups where relevant
- strict walk-forward backtesting across seasons
- metric aggregation across years

Metrics used in this project style:
- MAE for finish-position error
- top-k correctness trends (for ranking usefulness)
- calibration behavior under different blend weights

Tuning layers:
- model hyperparameters (estimators, depth, learning rate)
- calibration parameters (model_position_weight)
- optional simulation parameters for realism validation

### 6.4 Tools and libraries used
- Scikit-learn split and metric utilities
- custom tuning scripts under src/tuning
- optional Optuna workflow for hyperparameter search

### 6.5 Simple tuning example
Calibration sweep idea:
- test model_position_weight in [0.0, 0.05, 0.10, 0.15, 0.20]
- pick value with best weighted multi-year MAE

This links data-driven validation with operational default selection.

### 6.6 Workflow summary
Define split policy -> Run baseline -> Tune one layer at a time -> Re-evaluate walk-forward -> lock settings.

### 6.7 Practical challenges and assumptions
- Challenge: season shifts can invalidate previously best hyperparameters.
- Challenge: one metric may conflict with another objective.
- Assumption: weighted multi-season metrics represent deployment priorities.

### 6.8 How this connects to overall project
Part 6 determines whether Part 5 outputs are trustworthy enough to drive Part 7 simulation and Part 8 dashboard communication.

### 6.9 Part 6 key takeaways
- Temporal evaluation is mandatory in sequential sports contexts.
- Tune in controlled layers, not all parameters at once.
- Calibrated realism is part of model quality.

### 6.10 Part 6 revision points
- Remember why walk-forward is preferred.
- Remember difference between model tuning and calibration tuning.
- Remember to evaluate across multiple seasons, not one split.

### 6.11 Part 6 interview and viva questions
1. Why can random split overstate this project performance?
2. How do you justify calibration parameter choices scientifically?
3. What does it mean if MAE improves but winner hit-rate drops?
4. How would you design a robust evaluation report for stakeholders?
5. What tuning step would you freeze first for reproducibility?

### 6.12 Part 6 interview and viva answers
1. Random splits mix future and past race contexts, creating optimistic leakage-like evaluation.
2. Use walk-forward experiments, multi-season aggregation, and sensitivity analysis across candidate values.
3. The model improved average rank error but may have weakened top-end winner discrimination.
4. Include dataset scope, split method, multiple metrics, calibration behavior, and failure-case analysis.
5. Freeze calibration defaults and preprocessing contracts first, then tune model hyperparameters incrementally.

---

