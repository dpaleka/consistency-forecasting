import click
import os
from pathlib import Path
import json
from ground_truth_run import make_result_dict
from common.datatypes import ForecastingQuestion, Forecast
from static_checks import choose_checkers


def arbitragify(forecast_P: dict, metadata_checks: list[dict]) -> float:
    P_weight = 1.0
    P_prob = forecast_P["elicited_prob"]
    for check in metadata_checks:
        check_name = check["name"]
        check_data = check["data"]
        checker = choose_checkers([check_name])[check_name]
        hypocrite_answers = {
            "P": P_prob,
            **{q: v["elicited_prob"] for q, v in check_data.items()},
        }
        other = len(check_data)
        cons_answers, v = checker.max_min_arbitrage(
            hypocrite_answers, scoring=[P_weight] + [1.0] * other
        )
        P_weight += 1.0 * other
        P_prob = cons_answers["P"]
    return P_prob


def extract_intermediate_breadth_lines(line: dict) -> list[dict]:
    question = line["question"]
    metadata = line["forecast"]["metadata"]

    forecast_Ps = [m for m in metadata if m["name"] == "P"]
    assert len(forecast_Ps) == 1, "Expected 1 P, got {}".format(len(forecast_Ps))
    forecast_P = forecast_Ps[0]

    metadata_checks = [m for m in metadata if m["name"] != "P"]
    breadth = len(metadata_checks)

    intermediate_lines = []
    for i in range(breadth):
        intermediate_metadata = forecast_Ps + metadata_checks[:i]
        prob = arbitragify(forecast_P, metadata_checks[:i])
        intermediate_line = make_result_dict(
            line=line["question"],
            fq=ForecastingQuestion(**question),
            forecast=Forecast(prob=prob, metadata=intermediate_metadata),
        )
        intermediate_lines.append(intermediate_line)

    return intermediate_lines


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

    intermediate_liness_: list[list[dict]] = [
        extract_intermediate_breadth_lines(line) for line in lines
    ]
    intermediate_liness = [
        list(intermediate_lines) for intermediate_lines in zip(*intermediate_liness_)
    ]

    for i, intermediate_lines in enumerate(intermediate_liness):
        output_dir = input_dir + f"_{i}x"
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / "ground_truth_results.jsonl"
        with open(output_file, "w") as f:
            for line in intermediate_lines:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()

# python src/extract_intermediate_breadth_cf_calls.py --input_dir src/data/forecasts/recalc_test/groundtruth_broad
