## RQ1_F1_response_curves
RQ1. Severity response, with paired uncertainty

Curves are root mean squared errors pooled over corruption replicas relative to the same clean model. Bands: 95% replicate plus clock-day block bootstrap on the fixed historical test period. Nominal severity has mechanism-specific meaning. Datasets are not pooled.

## RQ1_F2_vulnerability_fingerprint
RQ1. Mechanism-specific vulnerability

Both panels use an identical color normalization. Negative areas are retained and indicate lower error in this stress setting, not universal superiority. All models, including naive baselines, are shown.

## RQ1_F3_injection_manipulation_check
RQ1. Was the requested missingness actually achieved?

Denominator is originally observed raw cells over every clock origin, not only origins with an observed future label. Channel rounding, protected MAR context, and conditional risk change realized rates. Bands are empirical corruption-replicate 2.5-97.5 percentiles, not confidence intervals.

## RQ1_T1_accuracy_and_degradation
RQ1. Absolute accuracy and relative degradation

Equal weights across the prespecified mechanisms, not learned weights. AUDC is normalized by the maximum severity. Worst scenario is the maximum pooled relative RMSE change. Clean RMSE units differ by dataset: Air mg/m3; Energy Wh per 10 minutes. Relative changes and AUDC are fractions (0.10 = 10%). Detailed paired block-sign tests and Holm adjustments are separate result files.

## RQ2_F1_response_curves
RQ2. Severity response, with paired uncertainty

Curves are root mean squared errors pooled over corruption replicas relative to the same clean model. Bands: 95% replicate plus clock-day block bootstrap on the fixed historical test period. Nominal severity has mechanism-specific meaning. Datasets are not pooled.

## RQ2_F2_vulnerability_fingerprint
RQ2. Mechanism-specific vulnerability

Both panels use an identical color normalization. Negative areas are retained and indicate lower error in this stress setting, not universal superiority. All models, including naive baselines, are shown.

## RQ2_F3_tail_error_exposure
RQ2. Typical error and tail-error exposure are different

Each point is a model/fault combination at the largest registered severity, averaged over replicas. This diagnostic distinguishes typical error from the tail of the error distribution; it is not an uncertainty interval or a worst-case guarantee.

## RQ2_T1_accuracy_and_degradation
RQ2. Absolute accuracy and relative degradation

Equal weights across the prespecified mechanisms, not learned weights. AUDC is normalized by the maximum severity. Worst scenario is the maximum pooled relative RMSE change. Clean RMSE units differ by dataset: Air mg/m3; Energy Wh per 10 minutes. Relative changes and AUDC are fractions (0.10 = 10%). Detailed paired block-sign tests and Holm adjustments are separate result files.

## RQ3_F1_rank_stability
RQ3. Stability compared with repeat-only explanation noise

Grouped permutation importance is signed. Clean repeat control changes permutation draws without corrupting the data. Other conditions use paired permutation draws. Both panels share [-1, 1]. Values are conditional on the fixed, contiguous explanation windows.

## RQ3_F2_signed_importance_changes
RQ3. Which evidence changes, and in which direction?

Deployment model is selected using development CV only. Displayed groups are selected by absolute clean importance; negative importance is preserved. All lags of a sensor are permuted together within contiguous evaluation windows. This is grouped permutation evidence, not SHAP or a causal explanation.

## RQ3_F3_prediction_explanation_dissociation
RQ3. Prediction and explanation can deteriorate differently

Each point is a prespecified corruption replica. The figure diagnoses dissociation; it is not a pooled independent-sample correlation test. Clean-repeat controls are reported separately.

## RQ3_T1_explanation_metrics
RQ3. Degradation beyond clean-repeat explanation noise

Every entry is a paired degraded-minus-clean-repeat contrast using the same model and replica. Negative rank or agreement contrasts indicate degradation beyond the explainer control. Intervals bootstrap paired replicas; p-values use paired sign flips and Holm correction within dataset/model/metric.

## RQ4_F1_calibration_feasibility
RQ4. Feasibility is a constraint, not a best-effort label

Rings mark provisional candidates; filled markers require the exact one-sided event-rate upper bound to meet the target. Zero observed alarms are not treated as proof of a sufficiently low future false-alarm rate.

## RQ4_F2_alarm_timelines
RQ4. Residual evolution and all alarms, including early alarms

Prespecified abrupt-mean replica 0. Alarm timestamps correspond to residual availability. Pre-injection alarms are not filtered out. Real chronological data have no verified natural-drift labels; these are alarms around a simulated measurement shift.

## RQ4_F3_non_oracle_adaptation
RQ4. Does deployable adaptation beat frozen and periodic baselines?

Every prediction precedes its outcome. The random policy receives the alarm policy's refit-attempt budget, separating timing value from compute value. Guarded candidates use trailing observed labels. Negative is better; comparison with periodic updating is also saved numerically.

## RQ4_T1_operating_characteristics
RQ4. Operating characteristics and matched clean control

Rates are averages over simulated corruptions on one historical test period. Post-onset alarm is not proof of causal detection; compare the same-window clean alarm rate. Residual lag counts updates after the first affected residual, not elapsed time from sensor onset; zero means the arrival of that residual. Misses have undefined lag, never zero. Full censoring information is saved separately.

## RQ5_F1_coverage_width
RQ5. Empirical coverage versus interval width

Both methods use the identical calibration-B observations after a temporal gap from local-scale fitting. Bands use replicate plus clock-day resampling. Serial dependence and distribution shift preclude an unconditional exchangeability guarantee; the dashed line is nominal, not guaranteed.

## RQ5_F2_tie_safe_risk_coverage
RQ5. Selective prediction without arbitrary tie breaking

Prespecified replica 0. Solid dots: attainable local-width thresholds. Cross at 100%: constant global width has no individual ranking. Dashed lines: sqrt(expected MSE) for random selection. Operational thresholds are fixed on policy data; these curves are descriptive, not test-tuned policies.

## RQ5_F3_environmental_decision_tradeoff
RQ5. Uncertainty changes environmental decision cost and review demand

High-event thresholds are frozen at the policy 90% quantile. The panel fixes abstention cost at 0.25, varies feasible review capacity, and uses a false-negative cost ratio of 5. The full cost/capacity grid is saved; these are scenarios, not stakeholder utilities.

## RQ5_T1_uncertainty_decision
RQ5. Calibration, warning, and decision utility are distinct endpoints

Deployment models only. Decision columns use the policy 90% event threshold and false-negative cost ratio 5; the complete prespecified sensitivity grid is in RQ5_decision_utility.csv. Costs are scenario assumptions, not inferred stakeholder preferences.

## RQ6_F1_accuracy_runtime_frontier
RQ6. Accuracy and measured training cost

Costs use one host, fixed model seeds, excluded warmups and randomized repeat order. Horizontal bars are the empirical 2.5-97.5 percentiles across timing repeats, not cross-hardware confidence intervals. The dotted nondominated set is computed from means only.

## RQ6_F2_energy_evidence_and_provenance
RQ6. Memory remains measurable when energy telemetry is unavailable

Missing telemetry remains unavailable, never zero or assumed model-specific wattage. CPU package counters are host-scoped; CodeCarbon fallback values are estimates. When all energy readings are unavailable this panel displays measured process peak RSS and increments, not a fabricated energy comparison. Process RSS includes prior allocations.

## RQ6_F3_training_inference_cost
RQ6. Training and inference are distinct operational costs

Inference is normalized per 1,000 predictions. Time covers computation, whereas energy covers the tracked operation batch and may also include instrumentation overhead. No arbitrary robustness-per-watt composite is formed.

## RQ6_T1_compute_and_measurement_scope
RQ6. Measured cost and energy qualification

Peak RSS is process resident memory, not model-only allocation. A zero RSS increment can mean previously allocated memory was reused. Energy estimates are not converted into carbon claims. Detailed operation counts, warnings and hardware records are included.

## EV_F1_multisite_skill
External replication across all 12 Beijing stations

Every point is a genuinely held-out station test result. LOSO models are trained on the other eleven stations with fixed hyperparameters. Values below one beat persistence; failures are retained.

## EV_F2_multisite_stress
Between-station distribution of degradation

Boxes summarize the twelve station-level mean effects; corruption replicates are averaged within station before cross-station display. Stations, not simulated replicas, are the generalisation unit.

## EV_F3_transfer_penalty
Cross-site transfer penalty is heterogeneous

Positive values mean that training on other stations transfers worse than station-local training. This isolates transportability rather than mixing stations with corruption replicas.

## EV_T1_multisite_endpoints
External multi-site replication endpoints

All twelve stations are shown. Hyperparameters are fixed before held-out-station testing. Model and persistence metrics use the same explicitly reported scoreable rows; targets lacking a persistence value are excluded only from this paired comparison and counted in the audit. Ratios below one beat persistence. The station bootstrap in BEIJING_multisite_summary.csv is descriptive and does not imply that Beijing stations represent all cities.
