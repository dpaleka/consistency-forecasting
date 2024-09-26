"""This script is run on `ground_truth_results.jsonl` files
to extract the intermediate CF calls, i.e. the ones that are not
at the depth of 4."""

import os
import click
import json
from pathlib import Path
from evaluation_utils.utils import write_to_dirs
from .ground_truth_run import make_result_dict
from common.datatypes import Forecast, ForecastingQuestion


def get_hypocrite_call(metadata: list) -> dict | None:
    """Get the immediate child's prob and metadata."""
    if (
        not isinstance(metadata, list)
        or "name" not in metadata[0]
        or len(metadata) == 0
    ):
        assert False, "this should not be called on a Basic forecast"
    hypocrite_items = [m for m in metadata if m.get("name", None) == "P"]
    assert len(hypocrite_items) == 1, "Expected 1 hypocrite item, got {}".format(
        len(hypocrite_items)
    )
    hypocrite_item = hypocrite_items[0]
    hypocrite_prob = hypocrite_item["elicited_prob"]
    hypocrite_metadata = hypocrite_item["elicitation_metadata"]
    return {"prob": hypocrite_prob, "metadata": hypocrite_metadata}


def get_intermediate_forecasts(metadata: list) -> list[dict]:
    """Get all calls recursively, in descending order of depth."""
    calls = []
    while metadata is not None:
        call = get_hypocrite_call(metadata)
        calls.append(call)
        metadata = call["metadata"]
    # depth = len(calls)
    return calls


@click.command()
@click.option(
    "--input_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the original RCF ground truth results",
)
def main(
    input_dir: str,
):
    input_file = Path(input_dir) / "ground_truth_results.jsonl"
    with open(input_file, "r") as f:
        lines = [json.loads(line) for line in f]
    questions = [line["question"] for line in lines]
    metadatas = [line["forecast"]["metadata"] for line in lines]
    callss = [get_intermediate_forecasts(metadata) for metadata in metadatas]
    intermediary_forecastss = [
        list(intermediary_forecasts) for intermediary_forecasts in zip(*callss)
    ]

    print(
        f"Depth: {len(intermediary_forecastss)}. Extracted as many intermediary forecasts."
    )

    output_dirs = [
        input_dir + f"_{depth}" for depth in range(len(intermediary_forecastss))
    ]
    for output_dir, intermediary_forecasts in zip(output_dirs, intermediary_forecastss):
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for question, forecast in zip(questions, intermediary_forecasts):
            line = {
                "question": question,
                "forecast": forecast,
            }
            result = make_result_dict(
                line=line,
                fq=ForecastingQuestion(**question),
                forecast=Forecast(**forecast),
            )
            results.append(result)
        write_to_dirs(
            results, "ground_truth_results.jsonl", [Path(output_dir)], overwrite=True
        )
        print(f"Wrote {len(results)} results to {output_dir}")


if __name__ == "__main__":
    main()
