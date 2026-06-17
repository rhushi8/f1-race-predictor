# Part 4: Feature Engineering

### 4.1 What this step is
Feature engineering transforms cleaned raw signals into predictive race intelligence.
In this project, features combine weekend pace, degradation behavior, historical form, and context.

### 4.2 Why it is important
Model performance in tabular motorsport data is strongly feature-dependent.
Better representations usually outperform naive model complexity increases.

### 4.3 How it works internally
Main feature families:
- qualifying features: q1/q2/q3 times, gap to pole, grid position
- pace features: best laps, long-run pace, fuel-corrected pace
- tire features: compound-specific degradation slopes and crossover heuristics
- driver/team context: ELO ratings, rolling form, DNF history
- circuit context: circuit affinity and circuit-level risk features
- weather/context features: air temp, track temp, humidity, wind, track evolution

Key engineering logic:
- compute pre-race historical aggregates
- avoid temporal leakage by using prior races only
- add interaction features such as pace and degradation combinations

### 4.4 Tools and libraries used
- Pandas and NumPy for aggregation and transforms
- project-specific feature pipeline in src/features/engineer.py

### 4.5 Simple dummy example
Raw columns:
- fp2_long_run_pace_s = 91.2
- deg_slope_medium = 0.08

Engineered interaction:
- pace_deg_product = 91.2 * 0.08 = 7.296

This feature can separate drivers with similar one-lap pace but different tire sustainability.

### 4.6 Workflow summary
Collect candidate signals -> Engineer domain features -> Validate temporal correctness -> Select final model columns.

### 4.7 Practical challenges, assumptions, limitations
- Assumption: historical behavior generalizes across nearby races.
- Challenge: regulation changes can reduce feature stability across years.
- Limitation: some signals are noisy proxies, not direct race strategy truth.

### 4.8 How this connects to overall project
Part 4 defines the information content given to XGBoost and LightGBM in Part 5.
It also sets inputs used for simulation profile realism in Part 7.

### 4.9 Part 4 key takeaways
- Feature quality often dominates algorithm choice in tabular prediction.
- Temporal leakage control is non-negotiable.
- Interaction terms encode racing dynamics beyond raw values.

### 4.10 Part 4 revision points
- Memorize major feature families and why each exists.
- Remember one concrete interaction feature and its intuition.
- Remember temporal cutoff rule for historical features.

### 4.11 Part 4 interview and viva questions
1. Which engineered features are most impactful for race prediction and why?
2. How do you detect leakage in historical feature creation?
3. Why use ELO-style features in this domain?
4. How would you adapt features for a major regulation change season?
5. What is one feature you would remove first during ablation and why?

### 4.12 Part 4 interview and viva answers
1. Qualifying gaps, long-run pace, degradation, and recent form are usually most impactful because they reflect race-ready performance.
2. Verify each historical feature uses only prior races with temporal split tests and walk-forward checks.
3. ELO captures evolving driver and team strength in a compact, updateable signal.
4. Increase recent-season weighting, add regime indicators, and revalidate feature stability quickly.
5. Remove the least stable high-missing proxy first and measure impact through controlled ablation.

---

