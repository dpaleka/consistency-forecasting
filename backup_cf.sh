#!/bin/bash

# Define the backup directory
backup_dir="../cf_backup_2"

# Create the backup directory if it doesn't exist
mkdir -p "$backup_dir"

# Loop through each directory in dirs_to_backup.txt
while IFS= read -r dir; do
  # Check if the directory exists
  if [ -d "$dir" ]; then
    echo "Backing up directory: $dir"
    # Copy the directory to the backup folder
    cp -r "$dir" "$backup_dir"
  else
    echo "Directory not found: $dir"
  fi
done < dirs_to_track.txt