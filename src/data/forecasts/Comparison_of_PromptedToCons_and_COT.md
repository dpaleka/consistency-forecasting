## Comparison of PromptedToCons and COT

### Ground Truths
| Metric | PromptedToCons | COT | Difference |
|--------|-----------------|-----|------------|
| total_questions | 100 | 242 | -142 |
| avg_brier_score | 0.24 | 0.227 | 0.013 |
| avg_platt_brier_score | 0.207 | 0.201 | 0.006 |
| tuned_brier_baseline | 0.228 | 0.227 | 0.001 |
| avg_brier_score_scaled | 4.1 | 9.3 | -5.2 |
| avg_platt_brier_score_scaled | 17.1 | 19.8 | -2.7 |
| tuned_brier_baseline_scaled | 9.0 | 9.3 | -0.3 |
| avg_log_score | 0.97 | 0.921 | 0.049 |
| platt_scaling_factor | 1.352 | 1.072 | 0.28 |
| brier_score_decomposition.uncertainty | 0.227 | 0.227 | 0.0 |
| brier_score_decomposition.reliability | 0.138 | 0.125 | 0.013 |
| brier_score_decomposition.resolution | 0.048 | 0.034 | 0.014 |
| calibration_error | 0.237 | 0.148 | 0.089 |

### Key Differences

- COT has a significantly higher number of total questions (242) compared to PromptedToCons (100).
- PromptedToCons has a higher average Brier score (0.25) compared to COT (0.227), indicating a lower overall performance.
- COT has a lower calibration error (0.148) compared to PromptedToCons (0.24), suggesting better calibration.
- PromptedToCons has a higher average log score (0.999) compared to COT (0.921), indicating better overall performance in terms of log scores.
- The Platt scaling factor is higher for PromptedToCons (1.308) compared to COT (1.072).
- The brier_score_decomposition.reliability is lower for PromptedToCons (0.096) compared to COT (0.125), suggesting better reliability for PromptedToCons.

Overall, the data suggests that COT has a higher number of questions and better calibration, while PromptedToCons has a slightly better average log score and reliability. However, it's important to consider the specific use case and requirements to determine which model performs better for a given task.



### Evaluation
| Metric | PromptedToCons | COT | Difference |
|--------|-----------------|-----|------------|
| default_avg_violation_NegChecker | 0.036021 | 0.032393 | 0.003628 |
| default_median_violation_NegChecker | 0.022816 | 0.012004 | 0.010812 |
| frequentist_avg_violation_NegChecker | 0.213857 | 0.190483 | 0.023374 |
| frequentist_median_violation_NegChecker | 0.214614 | 0.155043 | 0.059571 |
| default_scaled_avg_violation_NegChecker | 0.01801 | 0.016196 | 0.001814 |
| default_scaled_median_violation_NegChecker | 0.011408 | 0.006002 | 0.005406 |
| default_avg_violation_AndChecker | 0.010148 | 0.013978 | -0.00383 |
| default_median_violation_AndChecker | 0.0 | 0.0 | 0.0 |
| frequentist_avg_violation_AndChecker | 0.074483 | 0.087308 | -0.012825 |
| frequentist_median_violation_AndChecker | 0.0 | 0.0 | 0.0 |
| default_scaled_avg_violation_AndChecker | 0.003383 | 0.004659 | -0.001276 |
| default_scaled_median_violation_AndChecker | 0.0 | 0.0 | 0.0 |
| default_avg_violation_OrChecker | 0.001801 | 0.022114 | -0.020313 |
| default_median_violation_OrChecker | 0.0 | 0.0 | 0.0 |
| frequentist_avg_violation_OrChecker | 0.030928 | 0.111408 | -0.08048 |
| frequentist_median_violation_OrChecker | 0.0 | 0.0 | 0.0 |
| default_scaled_avg_violation_OrChecker | 0.0006 | 0.007371 | -0.006771 |
| default_scaled_median_violation_OrChecker | 0.0 | 0.0 | 0.0 |
| default_avg_violation_AndOrChecker | 0.010875 | 0.026598 | -0.015723 |
| default_median_violation_AndOrChecker | 0.001355 | 0.005422 | -0.004067 |
| frequentist_avg_violation_AndOrChecker | 0.116759 | 0.168982 | -0.052223 |
| frequentist_median_violation_AndOrChecker | 0.0 | 0.112438 | -0.112438 |
| default_scaled_avg_violation_AndOrChecker | 0.002725 | 0.006649 | -0.003924 |
| default_scaled_median_violation_AndOrChecker | 0.000339 | 0.001355 | -0.001016 |
| default_avg_violation_ButChecker | 0.134589 | 0.078065 | 0.056524 |
| default_median_violation_ButChecker | 0.127674 | 0.021617 | 0.106057 |
| frequentist_avg_violation_ButChecker | 0.608601 | 0.410046 | 0.198555 |
| frequentist_median_violation_ButChecker | 0.622126 | 0.377285 | 0.244841 |
| default_scaled_avg_violation_ButChecker | 0.033647 | 0.026022 | 0.007625 |
| default_scaled_median_violation_ButChecker | 0.031918 | 0.007206 | 0.024712 |
| default_avg_violation_CondChecker | 0.071002 | 0.092049 | -0.021047 |
| default_median_violation_CondChecker | 0.035478 | 0.040777 | -0.005299 |
| frequentist_avg_violation_CondChecker | 0.342989 | 0.429661 | -0.086672 |
| frequentist_median_violation_CondChecker | 0.301169 | 0.373437 | -0.072268 |
| default_scaled_avg_violation_CondChecker | 0.023667 | 0.023012 | 0.000655 |
| default_scaled_median_violation_CondChecker | 0.011826 | 0.010194 | 0.001632 |
| default_avg_violation_ConsequenceChecker | 0.00372 | 0.003443 | 0.000277 |
| default_median_violation_ConsequenceChecker | 0.0 | 0.0 | 0.0 |
| frequentist_avg_violation_ConsequenceChecker | 0.027407 | 0.031345 | -0.003938 |
| frequentist_median_violation_ConsequenceChecker | 0.0 | 0.0 | 0.0 |
| default_scaled_avg_violation_ConsequenceChecker | 0.00186 | 0.001721 | 0.000139 |
| default_scaled_median_violation_ConsequenceChecker | 0.0 | 0.0 | 0.0 |
| default_avg_violation_ParaphraseChecker | 0.007101 | 0.016015 | -0.008914 |
| default_median_violation_ParaphraseChecker | 0.002561 | 0.002854 | -0.000293 |
| frequentist_avg_violation_ParaphraseChecker | 0.06941 | 0.117398 | -0.047988 |
| frequentist_median_violation_ParaphraseChecker | 0.071538 | 0.075507 | -0.003969 |
| default_scaled_avg_violation_ParaphraseChecker | 0.003551 | 0.008008 | -0.004457 |
| default_scaled_median_violation_ParaphraseChecker | 0.00128 | 0.001427 | -0.000147 |
| default_avg_violation_CondCondChecker | 0.134589 | 0.092049 | 0.04254 |
| default_median_violation_CondCondChecker | 0.127674 | 0.040777 | 0.086897 |
| frequentist_avg_violation_CondCondChecker | 0.608601 | 0.429661 | 0.17894 |
| frequentist_median_violation_CondCondChecker | 0.622126 | 0.373437 | 0.248689 |
| default_scaled_avg_violation_CondCondChecker | 0.033647 | 0.023012 | 0.010635 |
| default_scaled_median_violation_CondCondChecker | 0.031918 | 0.010194 | 0.021724 |
| default_avg_violation_ExpectedEvidenceChecker | 0.017117 | 0.026458 | -0.009341 |
| default_median_violation_ExpectedEvidenceChecker | 0.0 | 0.00426 | -0.00426 |
| frequentist_avg_violation_ExpectedEvidenceChecker | 0.145583 | 0.22093 | -0.075347 |
| frequentist_median_violation_ExpectedEvidenceChecker | 0.083088 | 0.170748 | -0.08766 |
| default_scaled_avg_violation_ExpectedEvidenceChecker | 0.004279 | 0.006614 | -0.002335 |
| default_scaled_median_violation_ExpectedEvidenceChecker | 0.0 | 0.001065 | -0.001065 |
| aggregated_default_avg_violation | 0.04759959999999999 | 0.0382392 | 0.00936639999999999 |
| aggregated_frequentist_avg_violation | 0.22742569999999995 | 0.21066120000000002 | 0.016764499999999932 |
| aggregated_default_scaled_avg_violation | 0.015287700000000001 | 0.0124058 | 0.002881900000000001 |