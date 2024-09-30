## Brier Score vs Consistency
```
python src/plot_consistency_vs_brier.py --all --dataset newsapi --gt_metric avg_platt_brier_score -t frequentist -c avg_violation --remove_gt_outlier 0.25
```
-> [output_plots/aggregated_vs_avg_platt_brier_score_frequentist_avg_violation_newsapi.png](output_plots/aggregated_vs_avg_platt_brier_score_frequentist_avg_violation_newsapi.png)

```
python src/plot_consistency_vs_brier.py --all --dataset newsapi --gt_metric avg_brier_score -t default_scaled -c avg_violation --remove_gt_outlier 0.25
```
-> [output_plots/aggregated_vs_avg_brier_score_default_scaled_avg_violation_newsapi.png](output_plots/aggregated_vs_avg_brier_score_default_scaled_avg_violation_newsapi.png)


```
python src/plot_consistency_vs_brier.py --all --dataset scraped --gt_metric avg_platt_brier_score -t frequentist -c avg_violation --remove_gt_outlier 0.25
```
-> [output_plots/aggregated_vs_avg_platt_brier_score_frequentist_avg_violation_scraped.png](output_plots/aggregated_vs_avg_platt_brier_score_frequentist_avg_violation_scraped.png)

```
python src/plot_consistency_vs_brier.py --all --dataset scraped --gt_metric avg_brier_score -t default_scaled -c avg_violation --remove_gt_outlier 0.25
```
-> [output_plots/aggregated_vs_avg_brier_score_default_scaled_avg_violation_scraped.png](output_plots/aggregated_vs_avg_brier_score_default_scaled_avg_violation_scraped.png)


### Correlation Table
```
python src/create_correlation_table.py --dataset newsapi
```
-> [output_plots/correlation_table_newsapi.csv](output_plots/correlation_table_newsapi.csv)

```
python src/create_correlation_table.py --dataset scraped
```
-> [output_plots/correlation_table_scraped.csv](output_plots/correlation_table_scraped.csv)




