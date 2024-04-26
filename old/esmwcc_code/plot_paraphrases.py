"""
Write a plotting script that displays a plot of different probabilities for the same paraphrased question.

Input: a json file that is a list of dicts. Each dict is of the form:

```
    {
        "questions": [
            "Will there be at least one fatality from an offensive nuclear detonation in Pakistan by 2030, if there's an offensive detonation anywhere by then?",
            "If an offensive nuclear detonation occurs anywhere by 2030, will Pakistan experience a minimum of one fatality due to such an event?",
            "Will at least one fatality result from an offensive nuclear blast in Pakistan by 2030, provided an offensive detonation takes place anywhere during that time frame?",
            "Assuming an offensive nuclear detonation happens anywhere by 2030, will Pakistan witness at least one fatality as a consequence of this event?"
        ],
        "responses": [
            (irrelevant here)
        ],
        "answers": [
            [
                {
                    "parsed": true,
                    "result": 0.4
                },
                {
                    "parsed": true,
                    "result": 0.4
                },
                {
                    "parsed": true,
                    "result": 0.3
                }
            ],
            [
                {
                    "parsed": true,
                    "result": 0.4
                },
                {
                    "parsed": true,
                    "result": 0.1
                },
                {
                    "parsed": true,
                    "result": 0.2
                }
            ],
            [
                {
                    "parsed": true,
                    "result": 0.95
                },
                {
                    "parsed": true,
                    "result": 0.95
                },
                {
                    "parsed": true,
                    "result": 0.95
                }
            ],
            [
                {
                    "parsed": true,
                    "result": 0.15
                },
                {
                    "parsed": true,
                    "result": 0.15
                },
                {
                    "parsed": true,
                    "result": 0.2
                }
            ]
        ],
        "type": "paraphrase",
        "violation": 0.7999999999999999,
        "median": [
            0.4,
            0.2,
            0.95,
            0.15
        ],
        "std_devs": [
            0.04714045207910319,
            0.12472191289246473,
            1.1102230246251565e-16,
            0.023570226039551594
        ]
    },
```

The x-axis should be the probability of the first question. For each paraphrase, plot the probability of the three different paraphrases on the y-axis.


"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker
import numpy as np
from plot_utils import jitter, clean, soft_logit, get_dimensions

PLOT_TYPES = ["scatter", "kdeplot", "kernel_scatter"]
SCALES = ["linear", "logit"]

def get_args():
    def get_parser():
        parser = argparse.ArgumentParser(description='Plot examples of non-monotonicity.')
        parser.add_argument('-i', '--input', type=str, help='The JSON file to read.')
        parser.add_argument('-p', '--plot_type', type=str, help='The type of plot to make.', choices=PLOT_TYPES, default="kdeplot")
        parser.add_argument('-s', '--scale', type=str, help='The scale of the plot.', choices=SCALES, default="linear")
        return parser

    args = get_parser().parse_args()
    
    return args


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_probabilities(data, plot_type='scatter', scale='linear'):
    fig, ax = plt.subplots(figsize=get_dimensions())

    points = []
    for i, item in enumerate(data):
        x = item["median"][0]
        y = item["median"][1:]
        points_now = [(x, y_) for y_ in y]
        points.extend(points_now)
    
    points = clean(points)

    if scale == 'logit':
        points = [(soft_logit(x), soft_logit(y)) for x, y in points]

    x_jitter, y_jitter = jitter(points)

    if plot_type == 'scatter':
        ax.scatter(x=x_jitter, y=y_jitter, color="k")

    elif plot_type == 'kernel_scatter':
        values = np.vstack([x_jitter, y_jitter])
        kernel = stats.gaussian_kde(values)(values)
        sns.scatterplot(x=x_jitter, y=y_jitter, c=kernel, cmap="viridis", ax=ax)
        
    elif plot_type == 'kdeplot':
        sns.set_palette("tab10")
        sns.scatterplot(x=x_jitter, y=y_jitter,  ax=ax)
        #sns.kdeplot(x=x_jitter, y=y_jitter, ax=ax, levels=5, fill=True, alpha=0.6, cut=2)

    ax.set_xlabel('P(question)')
    ax.set_ylabel('P(paraphrase)')
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    #ax.legend()

    return fig


def save_plot(fig, args, plot_type, scale):
    Path(f'new_plots/').mkdir(parents=True, exist_ok=True)
    input_file = Path(args.input)
    fig.savefig(f'new_plots/{input_file.stem}_{plot_type}_{scale}.png', bbox_inches='tight')
    fig.savefig(f'new_plots/{input_file.stem}_{plot_type}_{scale}.pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    args = get_args()
    data = read_json_file(args.input)
    plot_type = args.plot_type
    scale = args.scale

    fig = plot_probabilities(data, plot_type=plot_type, scale=scale)

    save_plot(fig, args, plot_type=plot_type, scale=scale)

if __name__ == "__main__":
    main()
