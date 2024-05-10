# %%
import json
import sys

# Add the src/common directory to the system path
# This is scripts/recompute_ids.py
from pathlib import Path

this_dir = Path(__file__).parent
sys.path.append(str(this_dir.parent / "src"))
print(sys.path)

from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path

DRY_RUN = False


def change_forecasting_questions_in_file(file_path):
    print("Changing forecasting questions in file:", file_path)
    ret = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            recursively_change_forecasting_questions(data)
            ret.append(data)

    if DRY_RUN:
        print("Dry run, not writing to file")
    else:
        with open(file_path, "w") as file:
            for line in ret:
                file.write(json.dumps(line) + "\n")


def recursively_change_forecasting_questions(data):
    if isinstance(data, dict):
        if (
            "question_type" in data
            and "title" in data
            and "body" in data
            and "resolution_date" in data
        ):  # Simple check for ForecastingQuestion-like format that
            print("Found forecasting-question-like:", data["title"])
            if "id" not in data:
                print("No id, not changing anything")
            else:
                q = ForecastingQuestion(**data)
                if q.id != data["id"]:
                    print("ID does not match computed ID, changing")
                    data["id"] = q.id
                else:
                    print("ID unchanged")
        for value in data.values():
            recursively_change_forecasting_questions(value)
    elif isinstance(data, list):
        for item in data:
            recursively_change_forecasting_questions(item)


def find_and_change_forecasting_questions():
    data_path = get_data_path()
    for jsonl_file in Path(data_path).rglob("*.jsonl"):
        change_forecasting_questions_in_file(jsonl_file)


find_and_change_forecasting_questions()


# %%
