import json
from datetime import datetime

# Load the JSON data from the file
with open('/Users/ashen/Desktop/consistency-forecasting/scripts/QUESTIONS_CLEANED.json', 'r') as file:
    data = json.load(file)

# Filter out non-binary questions
binary_questions = [q for q in data if q.get('question_type') == 'binary']

# Get the current date
current_date = datetime.now()

# Format the resolution_date and filter out dates not within 100 years
filtered_questions = []
for question in binary_questions:
    if question.get('resolution_date'):
        # Parse the resolution_date string into a datetime object
        resolution_date = datetime.strptime(question['resolution_date'], "%Y-%m-%dT%H:%M:%SZ")
        # Check if the resolution_date is within 100 years from the current date
        if abs((resolution_date - current_date).days) <= 36525:  # 365.25 * 100, accounting for leap years
            # Format the datetime object into the desired string format
            question['resolution_date'] = resolution_date.strftime("%b %d %Y %I:%M%p")
            filtered_questions.append(question)

# Write the modified data back to the file
with open('/Users/ashen/Desktop/consistency-forecasting/scripts/QUESTIONS_CLEANED_MODIFIED.json', 'w') as file:
    json.dump(filtered_questions, file, indent=4)

print("File has been updated with binary questions, formatted resolution dates, and filtered by date range.")