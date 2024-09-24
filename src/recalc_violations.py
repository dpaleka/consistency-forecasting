"""Ideally this should go in scripts/postpatch_scripts but it needs an import
from src and I don't want to bother with that"""

import json
import click
from static_checks.Checker import choose_checkers


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--checker", type=str, help="Checker to use for the check.")
@click.option(
    "--metric",
    default=["default", "frequentist"],
    multiple=True,
    help="Metrics to use for the check.",
)
def recalculate_violations(
    input_file: str,
    output_file: str,
    checker: str,
    metric: list[str],
):
    """
    Recalculate violations for each entry in the JSONL file and save the updated entries to a new file.

    INPUT_FILE: Path to the input JSONL file.
    OUTPUT_FILE: Path to the output JSONL file with recalculated violations.
    """
    updated_data = []
    checker = choose_checkers([checker])[checker]
    # Load the JSONL file
    with open(input_file, "r") as infile:
        for line in infile:
            data = json.loads(line)

            # Extract relevant probabilities
            answers = {
                "P": data["line"]["P"]["forecast"]["prob"],
                "Q": data["line"]["Q"]["forecast"]["prob"],
                "P_and_Q": data["line"]["P_and_Q"]["forecast"]["prob"],
                "P_or_Q": data["line"]["P_or_Q"]["forecast"]["prob"],
            }

            # Recalculate the violation_data
            recalculated_violations = checker.check_from_elicited_probs(
                answers=answers, metric=list(metric)
            )

            # Update the data with new violation_data
            data["violation_data"] = recalculated_violations

            # Add updated entry to the list
            updated_data.append(data)

    # Write the updated data to the output file
    with open(output_file, "w") as outfile:
        for entry in updated_data:
            outfile.write(json.dumps(entry) + "\n")

    click.echo(f"Recalculated violations and saved to {output_file}")


if __name__ == "__main__":
    recalculate_violations()

# python recalc_violations.py data/forecasts/BasicForecaster_09-24-13-56/AndOrChecker.jsonl data/forecasts/BasicForecaster_09-24-13-56/AndOrChecker_recalc.jsonl --metric default --metric frequentist
