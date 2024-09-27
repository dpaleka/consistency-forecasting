"""This script is run on consistency evaluation directories
to extract the intermediate CF elicitations, i.e. the ones that are not
at the depth of 4."""

import click
import json
from pathlib import Path
from static_checks import choose_checkers, Checker


def get_hypocrite_elicitation(line: dict) -> dict:
    hypocrite_line = {}
    for p in line:
        if p in ["violation_data", "metadata"]:
            continue
        metadata = line[p]["forecast"]["metadata"]
        hypocrite_items = [m for m in metadata if m.get("name", None) == "P"]
        assert len(hypocrite_items) == 1, "Expected 1 hypocrite item, got {}".format(
            len(hypocrite_items)
        )
        hypocrite_item = hypocrite_items[0]
        answer = {
            "prob": hypocrite_item["elicited_prob"],
            "metadata": hypocrite_item["elicitation_metadata"],
        }
        hypocrite_line[p] = {"question": line[p]["question"], "forecast": answer}
    # hypocrite["violation_data"] = ...
    return hypocrite_line


def get_intermediate_elicitations(line: dict) -> list[dict]:
    elicitations = []
    while line["P"]["forecast"]["metadata"] is not None:
        line = get_hypocrite_elicitation(line)
        elicitations.append(line)
    return elicitations


@click.command()
@click.option("--input_dir", type=click.Path(exists=True), required=True)
@click.option(
    "-k",
    "--relevant_checks",
    multiple=True,
    default=["all"],
    help='Relevant checks to perform. In case of "all", all checkers are used.',
)
def main(input_dir: str, relevant_checks: list[str]):
    checkers: dict[str, Checker] = choose_checkers(relevant_checks, Path(input_dir))
    input_files = [
        Path(input_dir) / f"{checker_name}.jsonl" for checker_name in checkers
    ]
    for input_file in input_files:
        with open(input_file, "r") as f:
            lines = [json.loads(line)["line"] for line in f]

        elicitationss: list[list[dict]] = [
            get_intermediate_elicitations(line) for line in lines
        ]
        intermediate_liness: list[list[dict]] = [
            list(elicitations) for elicitations in zip(*elicitationss)
        ]

        print(
            f"Depth: {len(intermediate_liness)}. Extracted as many intermediate elicitations."
        )

        output_dirs = [
            input_dir + f"_{len(intermediate_liness)-1-depth}"
            for depth in range(len(intermediate_liness))
        ]
        for output_dir in output_dirs:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        for intermediate_line, output_dir in zip(intermediate_liness, output_dirs):
            output_file = Path(output_dir) / input_file.name
            with open(output_file, "w") as f:
                for line in intermediate_line:
                    json.dump({"line": line}, f)
                    f.write("\n")


if __name__ == "__main__":
    main()
