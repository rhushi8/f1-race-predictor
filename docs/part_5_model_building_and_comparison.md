# Part 5: Model Building and Comparison

### 5.1 What this step is
Model building learns the mapping from engineered features to outcome-related targets.
This project uses a stacked ensemble with XGBoost, LightGBM, and additional components.

### 5.2 Why it is important
No single model captures all race dynamics robustly.
Stacking combines complementary learners and often improves stability.

### 5.3 How it works internally
1. Base learners produce out-of-fold predictions.
2. Meta-learner (Ridge regression) learns from these base predictions.
3. Incident model estimates DNF/safety-car related probabilities.
4. Full model is refit and serialized with artifacts.

### 5.4 XGBoost vs LightGBM in this project context
XGBoost:
- robust regularization and stable performance on heterogeneous tabular signals
- often strong when feature interactions are complex and noisy

LightGBM:
- faster training on large tabular sets
- efficient leaf-wise growth can capture patterns with lower compute cost

Why both are suitable here:
- F1 features are mixed-scale, interaction-heavy tabular signals.
- Combining both diversifies inductive biases and improves ensemble resilience.

### 5.5 Tools and libraries used
- XGBoost for gradient-boosted regression component
- LightGBM for pace-oriented regression component
- Scikit-learn for KFold and meta-learner
- optional PyTorch fallback logic for neural component portability

### 5.6 Simple example workflow
Inputs:
- 20 drivers x engineered features

Base outputs:
- model A predicts positions
- model B predicts positions
- model C predicts positions

Meta-model input:
- 20 x 3 base prediction matrix

Meta output:
- calibrated continuous finish-position predictions per driver

### 5.7 Practical challenges and trade-offs
- Ensemble complexity increases explainability burden.
- Artifact management becomes critical for reproducible inference.
- Overfitting risk rises if folds or temporal boundaries are mismanaged.

### 5.8 How this connects to overall project
Part 5 provides deterministic baseline outputs consumed by calibration and Monte Carlo simulation in Part 7.

### 5.9 Part 5 key takeaways
- Stacking improves robustness by combining diverse model strengths.
- XGBoost and LightGBM are both strong choices for this tabular domain.
- Artifact portability is as important as raw model score.

### 5.10 Part 5 revision points
- Remember base-model and meta-model roles.
- Remember key XGBoost vs LightGBM differences in this context.
- Remember why incident probability modeling is separate.

### 5.11 Part 5 interview and viva questions
1. Why choose stacking instead of one tuned booster?
2. In this project, when might LightGBM be preferred operationally over XGBoost?
3. How do out-of-fold predictions reduce leakage in stacking?
4. What artifact fields are essential for inference reproducibility?
5. How would you simplify this model for low-latency deployment?

### 5.12 Part 5 interview and viva answers
1. Stacking combines complementary model biases and improves robustness versus a single learner.
2. LightGBM is preferred when faster training or repeated tuning cycles are operational priorities.
3. Out-of-fold predictions ensure meta-learner inputs come from data not seen by that base model during training.
4. Feature column order, encoders, scaler, model metadata, and training version identifiers are essential.
5. Use one boosted model, reduce feature set, and lower tree complexity with calibration retained.

---

