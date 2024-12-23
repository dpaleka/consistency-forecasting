import re
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_entry_params(entry):
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - id: (\d+), url: (https?://\S+), title: (.+?), samples: (\d+), adv_prediction: (\d+\.\d+)%, basic_prediction: (\d+\.\d+)%, meta_prediction: (\d+\.\d+)%, submission: (\d+\.\d+)%,"
    match = re.search(pattern, entry)
    if match:
        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        entry_id = int(match.group(2))
        url = match.group(3)
        title = match.group(4)
        samples = int(match.group(5))
        adv_prediction = float(match.group(6)) / 100
        basic_prediction = float(match.group(7)) / 100
        meta_prediction = float(match.group(8)) / 100
        submission = float(match.group(9)) / 100
        return {
            "timestamp": timestamp,
            "id": entry_id,
            "url": url,
            "title": title,
            "samples": samples,
            "adv_prediction": adv_prediction,
            "basic_prediction": basic_prediction,
            "meta_prediction": meta_prediction,
            "submission": submission,
            "comments": [],
        }
    return None


def remove_errors(entries):
    cleaned_entries = []
    for entry in entries:
        if entry["comments"] != ["Comment Generation Error"]:
            cleaned_entries.append(entry)
    return cleaned_entries


def check_question_status(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        resolution_div = soup.find("div", {"class": "question-resolution"})
        if resolution_div:
            is_resolved = True
            resolution_text = resolution_div.find("p").text.strip()
            resolution_value = float(
                resolution_div.find("span", {"class": "resolution-value"}).text.strip()
            )
            return is_resolved, resolution_text, resolution_value
        else:
            return False, None, None
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while checking question status: {e}")
        return False, None, None


def process_log(log_file):
    entries = []
    current_entry = {}
    with open(log_file, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("2024-"):
                if current_entry:
                    entries.append(current_entry)
                params = extract_entry_params(line)
                if params:
                    current_entry = params
            elif current_entry:
                current_entry["comments"].append(line)
        if current_entry:
            entries.append(current_entry)
    return remove_errors(entries)


def scrape_resolution_info(df):
    for index, row in df.iterrows():
        url = row["url"]
        is_resolved, resolution_text, resolution_value = check_question_status(url)
        df.at[index, "is_resolved"] = is_resolved
        df.at[index, "resolution_text"] = resolution_text
        df.at[index, "resolution_value"] = resolution_value
    return df


# Example usage
processed_entries = process_log("downloaded_logs/submissions.log")
df = pd.DataFrame(processed_entries).drop("comments", axis=1)
# df = scrape_resolution_info(df)
print(df.T)
