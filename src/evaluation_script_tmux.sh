#!/bin/bash

# Check if the first argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <simple|full>"
  exit 1
fi

# Determine the forecaster based on the argument
if [ "$1" == "simple" ]; then
  forecaster_command="-f BasicForecaster"
  config_command=""
  async_command="--async"
elif [ "$1" == "full" ]; then
  forecaster_command="-f AdvancedForecaster"
  config_command=" -c forecasters/forecaster_configs/default_gpt4o.yaml"
  async_command=""
else
  echo "Invalid argument. Use 'simple' or 'full'."
  exit 1
fi

# List of values for foo
foos=("NegChecker" "AndOrChecker" "ConsequenceChecker" "ParaphraseChecker" "CondChecker" "CondCondChecker" "AndChecker" "OrChecker" "ButChecker")

# Create a new tmux session named 'fc'
tmux new-session -d -s fc

# Iterate over each value in the list
for foo in "${foos[@]}"
do
  # Create a new tmux window named after the current value of foo within the 'fc' session
  tmux new-window -t fc -n "$foo" 
  
  # Send commands to activate conda environment
  tmux send-keys -t fc:$foo "conda activate consistency" C-m

  # Send the command to run the evaluation
  tmux send-keys -t fc:$foo "\
NO_CACHE=True \
MAX_CONCURRENT_QUERIES=5 \
python evaluation.py \
$forecaster_command \
$config_command \
$async_command \
-n 50 \
--run \
-k $foo \
| tee eval_${foo}.log \
" C-m
done

# Attach to the tmux session
tmux attach-session -t fc
