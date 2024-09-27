#!/bin/bash

SESSION="instantiate"
PANE="0"

while true; do
    # Get the current command running in the pane
    CURRENT_CMD=$(tmux list-panes -t ${SESSION} -F "#{pane_active} #{pane_pid} #{pane_current_command}" | grep "^1" | awk '{print $3}')

    # Check if there's an active command running in the pane
    if [ -z "$CURRENT_CMD" ] || [ "$CURRENT_CMD" == "bash" ]; then
        # If there's no command running, or it's just the shell prompt, run the Git commands
        git add .
        git commit -m "done running."
        git push origin -u instantiate_2028_synths
        break
    fi

    sleep 1
done