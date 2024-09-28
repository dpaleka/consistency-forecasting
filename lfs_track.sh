#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <directory> [--migrate]"
    exit 1
fi

# Set variables
INPUT_DIR=$1
MIGRATE_FLAG=$2

# Find all files over 10MB in the input directory
FILES=$(find "$INPUT_DIR" -type f -size +10M)

# If no files are found, exit the script
if [ -z "$FILES" ]; then
    echo "No files larger than 10MB found in the directory."
    exit 0
fi

# Print the list of files found and ask for confirmation
echo "The following files larger than 10MB were found:"
echo "$FILES"
echo
read -p "Do you want to proceed with tracking these files with Git LFS? (y/n) " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Operation canceled by the user."
    exit 0
fi

# Track files with Git LFS
echo "$FILES" | while read -r FILE; do
    git lfs track "$FILE"
done

# Commit .gitattributes to the repository
git add .gitattributes
git commit -m "Tracking files in $INPUT_DIR with Git LFS"

# Check if the --migrate flag is set
if [ "$MIGRATE_FLAG" == "--migrate" ]; then
    # Convert the list of files to a comma-separated format
    FILES_COMMA_SEPARATED=$(echo "$FILES" | paste -sd, -)
    
    # Run git lfs migrate import on the list of files
    git lfs migrate import --include="$FILES_COMMA_SEPARATED"
fi

echo "Operation completed successfully."
