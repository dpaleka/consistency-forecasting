Consistent Forecasters configs:
- CF-4xEE1: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1`
- CF-N4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker]' -o depth=4`
- CF-P4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ParaphraseChecker]' -o depth=4`
- CF-NP4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker, ParaphraseChecker]' -o depth=4`

I will also create the results for **_intermediate_ forecasters** (basically replacing each breadth or depth of 4 with 1, 2, 3) from these.

**best to recompute stats and plots for everything** -- this won't take much time, and will make sure everything is using the latest versions of violations etc

# Main runs

## CF-4xEE1

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831](src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_scraped](src/data/forecasts/ConsistentForecaster_4xEE1_scraped)

- [x] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi)

- [x] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped)

## CF-N4

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_20240701_20240831](src/data/forecasts/ConsistentForecaster_N4_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_scraped](src/data/forecasts/ConsistentForecaster_N4_scraped)

- [x] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi)

- [x] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_scraped](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped)

## CF-P4

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped](src/data/forecasts/ConsistentForecaster_P4_scraped)

- [x] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi)

- [x] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped)

## CF-NP4

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped](src/data/forecasts/ConsistentForecaster_NP4_scraped)

- [ ] consistency evaluation - newsAPI

-> running in: [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi)

- [ ] consistency evaluation - scraped

-> running in: [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped)

-----

# Intermediate runs

There are four scripts of relevance here. Example usage:
- `python src/extract_intermediate_depth_cf_calls.py --input_dir src/data/forecasts/recalc_test/groundtruth`
- `python src/extract_intermediate_breadth_cf_calls.py --input_dir src/data/forecasts/recalc_test/groundtruth_4xee1`
- `python src/extract_intermediate_depth_cf_elicitations.py --input_dir src/data/forecasts/recalc_test/eval_small_np4`
- `python src/extract_intermediate_breadth_cf_elicitations.py --input_dir src/data/forecasts/recalc_test/eval_small_np4`

## CF-3xEE1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_3x](src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_3x)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_scraped_3x](src/data/forecasts/ConsistentForecaster_4xEE1_scraped_3x)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_3x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_3x)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_3x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_3x)


## CF-2xEE1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_2x](src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_2x)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_scraped_2x](src/data/forecasts/ConsistentForecaster_4xEE1_scraped_2x)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_2x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_2x)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_2x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_2x)

## CF-1xEE1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_1x](src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_1x)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_scraped_1x](src/data/forecasts/ConsistentForecaster_4xEE1_scraped_1x)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_1x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_1x)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_1x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_1x)


## CF-0xEE1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_0x](src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_0x)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_scraped_0x](src/data/forecasts/ConsistentForecaster_4xEE1_scraped_0x)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_0x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_0x)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_0x](src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_0x)

## CF-N3

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_3](src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_3)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_scraped_3](src/data/forecasts/ConsistentForecaster_N4_scraped_3)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_3](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_3)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_3](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_3)

## CF-N2

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_2](src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_2)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_scraped_2](src/data/forecasts/ConsistentForecaster_N4_scraped_2)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_2](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_2)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_2](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_2)

## CF-N1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_1](src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_1)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_scraped_1](src/data/forecasts/ConsistentForecaster_N4_scraped_1)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_1](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_1)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_1](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_1)

## CF-N0

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_0](src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_0)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_scraped_0](src/data/forecasts/ConsistentForecaster_N4_scraped_0)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_0](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_0)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_0](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_0)

## CF-P3

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_3](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_3)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped_3](src/data/forecasts/ConsistentForecaster_P4_scraped_3)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_3](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_3)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_3](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_3)

## CF-P2

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_2](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_2)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped_2](src/data/forecasts/ConsistentForecaster_P4_scraped_2)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_2](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_2)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_2](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_2)

## CF-P1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_1](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_1)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped_1](src/data/forecasts/ConsistentForecaster_P4_scraped_1)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_1](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_1)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_1](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_1)

## CF-P0

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_0](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_0)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped_0](src/data/forecasts/ConsistentForecaster_P4_scraped_0)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_0](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_0)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_0](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_0)

## CF-NP3

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_3](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_3)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped_3](src/data/forecasts/ConsistentForecaster_NP4_scraped_3)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_3](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_3)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_3](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_3)

## CF-NP2

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_2](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_2)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped_2](src/data/forecasts/ConsistentForecaster_NP4_scraped_2)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_2](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_2)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_2](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_2)
## CF-NP1

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_1](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_1)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped_1](src/data/forecasts/ConsistentForecaster_NP4_scraped_1)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_1](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_1)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_1](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_1)

## CF-NP0

- [ ] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_0](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_0)

- [ ] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped_0](src/data/forecasts/ConsistentForecaster_NP4_scraped_0)

- [ ] consistency evaluation - newsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_0](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_0)

- [ ] consistency evaluation - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_0](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_0)

-----

# Recalculation scripts

E.g.
```
python src/ground_truth_run.py --load_dir src/data/forecasts/recalc_test/groundtruth/
python src/evaluation.py --load_dir src/data/forecasts/recalc_test/eval_small_np4 -k all
python src/evaluation.py --load_dir src/data/forecasts/recalc_test/eval_n4 -k all
```
Just run these on all your directories. E.g.

```bash
ground_truth_directories=(
    "src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831"
    "src/data/forecasts/ConsistentForecaster_N4_20240701_20240831"
    "src/data/forecasts/ConsistentForecaster_P4_20240701_20240831"
    "src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831"
    "src/data/forecasts/ConsistentForecaster_4xEE1_scraped"
    "src/data/forecasts/ConsistentForecaster_N4_scraped"
    "src/data/forecasts/ConsistentForecaster_P4_scraped"
    "src/data/forecasts/ConsistentForecaster_NP4_scraped"
) # TODO: need to add the intermediate ones
evaluation_directories=(
    "src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi"
    "src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi"
    "src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi"
    "src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi"
    "src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped"
    "src/data/forecasts/ConsistentForecaster_N4_tuples_scraped"
    "src/data/forecasts/ConsistentForecaster_P4_tuples_scraped"
    "src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped"
) # TODO: need to add the intermediate ones

for DIR in "${ground_truth_directories[@]}"; do
    python src/ground_truth_run.py --load_dir "$DIR"
done

for DIR in "${evaluation_directories[@]}"; do
    python src/evaluation.py --load_dir "$DIR" -k all
done
```