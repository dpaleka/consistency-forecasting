import os
from static_checks.Checker import Checker, choose_checkers
from extract_intermediate_breadth_cf_calls import extract_intermediate_breadth_lines
import click
from pathlib import Path
import json


@click.command()
@click.option(
    "--input_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the original RCF elicitations",
)
@click.option(
    "-k",
    "--relevant_checks",
    multiple=True,
    default=["all"],
    help='Relevant checks to perform. In case of "all", all checkers are used.',
)
def main(input_dir: str, relevant_checks: list[str]):
    print(f"Starting process with input directory: {input_dir}")
    checkers: dict[str, Checker] = choose_checkers(relevant_checks, Path(input_dir))
    input_files = [
        Path(input_dir) / f"{checker_name}.jsonl" for checker_name in checkers
    ]
    for input_file in input_files:
        print(f"Processing file: {input_file}")
        with open(input_file, "r") as f:
            lines = [json.loads(line)["line"] for line in f]

        # intermediate_liness__ = [
        #     {
        #         p: extract_intermediate_breadth_lines(line_p, include_ground_truth_info=False)
        #         for p, line_p in line.items() if p not in ["violation_data", "metadata"]
        #     }
        #     for line in lines
        # ]
        intermediate_liness__ = []
        for line in lines:
            l = {}
            for p, line_p in line.items():
                if p not in [
                    "violation_data",
                    "metadata",
                    "prob",
                    "resolution",
                    "log_score",
                    "brier_score",
                    "brier_score_scaled",
                ]:
                    # print(p)
                    # print(line_p.keys())
                    l[p] = extract_intermediate_breadth_lines(
                        line_p, include_ground_truth_info=False
                    )
            intermediate_liness__.append(l)
        intermediate_liness_: list[list[dict]] = []
        for line_ in intermediate_liness__:
            # line_ = {'a': [1, 2, 3], 'b': [4, 5, 6]}
            # line = [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
            line = [
                {key: value for key, value in zip(line_.keys(), values)}
                for values in zip(*line_.values())
            ]
            intermediate_liness_.append(line)

        intermediate_liness = [
            list(intermediate_lines)
            for intermediate_lines in zip(*intermediate_liness_)
        ]

        print(f"Breadth: {len(intermediate_liness)}")

        for i, intermediate_lines in enumerate(intermediate_liness):
            output_dir = input_dir + f"_{i}x"
            os.makedirs(output_dir, exist_ok=True)
            output_file = Path(output_dir) / f"{input_file.stem}.jsonl"
            print(f"Writing output to: {output_file}")
            with open(output_file, "w") as f:
                for line in intermediate_lines:
                    f.write(json.dumps(line) + "\n")

    print("Process completed.")


if __name__ == "__main__":
    main()
