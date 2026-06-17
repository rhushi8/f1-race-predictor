# Part 10: Interview Notes and Viva Questions

### 10.1 60-second project summary (interview version)
I built an end-to-end Formula 1 race prediction system using Python, FastF1/OpenF1 data ingestion, feature engineering, stacked gradient boosting models, and Monte Carlo simulation for uncertainty-aware outcomes. The pipeline predicts win, podium, and points probabilities and serves results through an interactive Dash and Plotly dashboard. I focused on temporal correctness, artifact consistency, and realistic probabilistic output rather than only raw model error.

### 10.2 Whiteboard architecture answer template
1. State problem and outputs.
2. Explain ingestion sources and merge strategy.
3. Explain preprocessing and feature families.
4. Explain model stack and calibration.
5. Explain simulation loop and probability aggregation.
6. Explain dashboard and decision support outputs.
7. Explain reliability controls and limitations.

### 10.3 Frequently asked technical questions with short answer pointers
Q1: Why XGBoost and LightGBM?
- Both are strong tabular learners with complementary training behavior and bias profiles.

Q2: Why simulation after model prediction?
- Model gives central tendency; simulation gives uncertainty distribution and event sensitivity.

Q3: How do you prevent leakage?
- Enforce prior-only historical aggregates and walk-forward validation.

Q4: Why not deep learning end to end?
- Dataset structure and size favor boosted trees plus domain engineering for this use case.

Q5: How do you ensure reproducibility?
- Stable seeds, versioned artifacts, deterministic schema contracts, and consistent preprocessing.

### 10.4 Mini viva bank (practice)
1. Explain this project to a non-technical stakeholder in 30 seconds.
2. Distinguish model confidence from simulation uncertainty.
3. Describe one ingestion failure and your mitigation strategy.
4. Describe one preprocessing bug that can silently break predictions.
5. Explain one feature that captures racecraft better than raw pace.
6. Explain why walk-forward matters in motorsport forecasting.
7. Describe how you would test dashboard reliability.
8. Describe a rollback plan after a bad model release.
9. Explain how calibration parameters are selected.
10. Explain one realistic improvement for next season.

### 10.4A Mini viva answers
1. It predicts race outcome probabilities from pre-race data and explains them through simulation and dashboard views.
2. Model confidence is certainty of model estimate; simulation uncertainty is distribution spread from race events.
3. Example: missing OpenF1 session key; mitigation is fallback to FastF1 baseline with warning logs.
4. Refitting encoders at inference changes category mapping and silently corrupts model inputs.
5. Rolling form plus degradation interaction captures consistency and tire management better than one-lap pace.
6. It preserves time order, preventing optimistic evaluation from future data leakage.
7. Run schema tests, callback smoke tests, and fallback-mode rendering checks.
8. Revert to previous model artifact, rerun smoke predictions, and verify key metrics before reopening.
9. Through walk-forward calibration sweeps and multi-season weighted performance checks.
10. Add automated drift monitoring with season-segment alerts tied to retraining triggers.

### 10.5 Presentation tips for this project
- Start with decision question, not algorithm list.
- Show one clear pipeline diagram.
- Use one example race to demonstrate uncertainty outputs.
- Acknowledge limitations early to build credibility.
- End with prioritized roadmap, not vague future ideas.

### 10.6 Part 10 key takeaways
- Interview success depends on clarity, sequencing, and trade-off awareness.
- Demonstrating reliability thinking is as important as modeling skill.
- This project is strongest when explained as a full system, not isolated models.

### 10.7 Part 10 revision points
- Memorize 60-second summary and whiteboard flow.
- Prepare at least five short, technical Q&A responses.
- Keep one strong example for uncertainty-aware prediction explanation.

### 10.8 Part 10 interview and viva questions
1. What part of this pipeline would you productionize first and why?
2. If you had two weeks, what upgrades would produce highest impact?
3. Which metric would you report to technical vs non-technical audiences?
4. What failure mode worries you most before race weekend?
5. How would you defend model decisions under uncertainty?

### 10.9 Part 10 interview and viva answers
1. Productionize ingestion and feature contracts first because they anchor every downstream output.
2. Add leakage tests and drift monitoring first, then improve calibration and release automation.
3. Technical: MAE and calibration diagnostics; non-technical: win probability ranges and confidence framing.
4. Schema drift or missing upstream data causing silent feature corruption is the highest concern.
5. Use probabilistic outputs, confidence intervals, validation evidence, and explicit assumptions.

---

End of teaching module.

