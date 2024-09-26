Consistent Forecasters configs:
- CF-4xEE1: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker, ExpectedEvidenceChecker]' -o depth=1`
- CF-N4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker]' -o depth=4`
- CF-P4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[ParaphraseChecker]' -o depth=4`
- CF-NP4: `-f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker, ParaphraseChecker]' -o depth=4`

I will also create the results for **_intermediate_ forecasters** (basically replacing each breadth or depth of 4 with 1, 2, 3) from these.

**best to recompute stats and plots for everything** -- this won't take much time, and will make sure everything is using the latest versions of violations etc

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

- [ ] consistency evaluation - newsAPI

-> running in: [src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi)

- [ ] consistency evaluation - scraped

-> running in:[src/data/forecasts/ConsistentForecaster_N4_tuples_scraped](src/data/forecasts/ConsistentForecaster_N4_tuples_scraped)

## CF-P4

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_P4_20240701_20240831](src/data/forecasts/ConsistentForecaster_P4_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_P4_scraped](src/data/forecasts/ConsistentForecaster_P4_scraped)

- [ ] consistency evaluation - newsAPI

-> will run in: [src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi)

- [ ] consistency evaluation - scraped

-> will run in: [src/data/forecasts/ConsistentForecaster_P4_tuples_scraped](src/data/forecasts/ConsistentForecaster_P4_tuples_scraped)

## CF-NP4

- [x] ground truth run - NewsAPI

-> [src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831](src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831)

- [x] ground truth run - scraped

-> [src/data/forecasts/ConsistentForecaster_NP4_scraped](src/data/forecasts/ConsistentForecaster_NP4_scraped)

- [ ] consistency evaluation - newsAPI

-> will run in: [src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi](src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi)

- [ ] consistency evaluation - scraped

-> will run in: [src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped](src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped)