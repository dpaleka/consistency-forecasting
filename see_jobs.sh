# DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped"
# DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi"
# DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped"
# DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi"
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped"
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi"

for f in $DIR/*.jsonl; do
    stat "$f"
done

wc -l $DIR/*.jsonl