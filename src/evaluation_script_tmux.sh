#!/bin/bash

# Check if the -t, -p, or -n argument is provided
while getopts ":t:p:n:" opt; do
  case $opt in
    t)
      type_arg=$OPTARG
      ;;
    p)
      tuple_dir=$OPTARG
      ;;
    n)
      num_lines=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 -t <simple|full> -p <tuple_dir> -n <num_lines>"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 -t <simple|full> -p <tuple_dir> -n <num_lines>"
      exit 1
      ;;
  esac
done


# Check if the type argument is set
if [ -z "$type_arg" ]; then
  echo "Usage: $0 -t <simple|full>"
  exit 1
fi

# Determine the forecaster based on the argument
if [ "$type_arg" == "simple" ]; then
  forecaster_command="-f BasicForecaster"
  config_command=""
  async_command="--async"
elif [ "$type_arg" == "full" ]; then
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
--tuple_dir $tuple_dir \
$forecaster_command \
$config_command \
$async_command \
--num_lines $num_lines \
--run \
-k $foo \
| tee eval_${foo}.log \
" C-m
done

# Attach to the tmux session
tmux attach-session -t fc
