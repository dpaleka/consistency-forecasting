# each is done once it hits 2001 and also have all the stats files in them
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi" # last: 1651 # paused, rerunning
DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_rerun" # last: 0
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped" # last: 201
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi" # last: 0

for f in $DIR/*.jsonl; do
    stat "$f"
done

wc -l $DIR/*.jsonl