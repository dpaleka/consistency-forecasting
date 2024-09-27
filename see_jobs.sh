#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./see_jobs.sh [scraped|newsapi]"
    exit 1
fi

# Set the directory based on the argument
if [ "$1" == "scraped" ]; then
    DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped" # last: 601
elif [ "$1" == "newsapi" ]; then
    DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi" # last: 451
else
    echo "Invalid argument. Use 'scraped' or 'newsapi'."
    exit 1
fi

# Run the commands on the selected directory
for f in $DIR/*.jsonl; do
    stat "$f"
done

# Find the earliest and latest modification times
earliest=$(stat -c %Y $DIR/*.jsonl | sort -n | head -1)
latest=$(stat -c %Y $DIR/*.jsonl | sort -n | tail -1)

# Convert timestamps to human-readable format
earliest_date=$(date -d @$earliest)
latest_date=$(date -d @$latest)

# Calculate the time difference in seconds and convert to human-readable form
time_running_seconds=$((latest - earliest))
time_running=$(date -u -d @$time_running_seconds +%H:%M:%S)

echo "Earliest file time: $earliest_date"
echo "Latest file time: $latest_date"
echo "Time Running: $time_running"

wc -l $DIR/*.jsonl

# each is done once it hits 2001 and also have all the stats files in them
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped" # DONE
# # DIR="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi" # DONE
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped" # last: 751
# DIR="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi" # last: 601

# for f in $DIR/*.jsonl; do
#     stat "$f"
# done

# wc -l $DIR/*.jsonl