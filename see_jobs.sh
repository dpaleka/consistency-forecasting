# each is done once it hits 2001 and also have all the stats files in them
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped" # DONE
# DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi" # last: 1151
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped" # NOT STARTED
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi" # NOT STARTED

for f in $DIR/*.jsonl; do
    stat "$f"
done

wc -l $DIR/*.jsonl