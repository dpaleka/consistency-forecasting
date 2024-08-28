from dataclasses import dataclass
from simple_parsing import ArgumentParser
import json

parser = ArgumentParser()


@dataclass
class Options:
    filename: str  # Filename to apply the conversion to


args = parser.add_arguments(Options, dest="opt")
args = parser.parse_args()
args = args.opt


def convert_format(format_1):
    format_2 = {
        "id": str(format_1["id"]),
        "title": format_1["title"],
        "body": format_1["body"],
        "data_source": format_1["data_source"],
        "created_date": format_1.get("created_date", None),
        "question_type": format_1["question_type"].lower(),
        "resolution_date": format_1["resolution_date"],
        "url": format_1.get("url", None),
        "metadata": format_1["metadata"],
        "resolution": format_1["resolution"],
    }
    # Fix body into resolution criteria and metadata.
    if isinstance(format_2["body"], dict) and "resolution_criteria" in format_2["body"]:
        format_2["metadata"]["background_info"] = format_2["body"]["background_info"]
        format_2["body"] = format_2["body"]["resolution_criteria"]
        assert isinstance(format_2["body"], str)

    if isinstance(format_1["resolution"], float) or isinstance(
        format_1["resolution"], int
    ):
        format_2["resolution"] = bool(format_2["resolution"])

    return format_2


with open(args.filename, "r") as f:
    if args.filename.endswith(".json"):
        data = json.load(f)
    elif args.filename.endswith(".jsonl"):
        data = [json.loads(line) for line in f]

# write jsonl
out_filename = args.filename.replace(".json", ".jsonl")
with open(out_filename, "w") as f:
    for datum in data:
        f.write(json.dumps(convert_format(datum)) + "\n")
