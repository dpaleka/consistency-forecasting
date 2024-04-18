"""
Write a plotting script that displays a plot of pairs of (question, negation question) probabilities.

Input: a json file that is a list of dicts. Each dict is of the form:

{
    "questions": [
      "Will a human made spaceship enter the Venusian atmosphere before 2030?",
      "Will no human made spaceship enter the Venusian atmosphere before 2030?"
    ],
    "responses": [
      [
        … not relevant for plotting
      ]
    ],
    "answers": [
      [
        {
          "parsed": true,
          "result": 0.65
        },
        {
          "parsed": true,
          "result": 0.6
        },
        {
          "parsed": true,
          "result": 0.8
        }
      ],
      [
        {
          "parsed": true,
          "result": 0.85
        },
        {
          "parsed": true,
          "result": 0.95
        },
        {
          "parsed": true,
          "result": 0.8
        }
      ]
    ],
    "type": "negated_pair",
    "violation": 0.5,
    "median": [
      0.65,
      0.85
    ],
    "std_devs": [
      0.08498365855987977,
      0.062360956446232324
    ]
  },

The plot should have x-axis probability for the first question, and y-axis probability for the second question. 
The plot should have a line x+y=1, and the points should be plotted on the plot. 

Use argparse like this:
```
def get_args():
    def get_parser():
        parser = argparse.ArgumentParser(description='Plot examples of non-monotonicity.')
        parser.add_argument('-i', '--input', type=str, help='The JSON file to read.')
        return parser

    args =  get_parser().parse_args(["--input", "top_experiments/sorted_violations/negation_pair_dataset_gpt-4-0314_method_1shot_china_T_0.0_times_3_mt_400.json"]) \
        if 'ipykernel' in sys.modules \
        else get_parser().parse_args()
    
    return args
``` 

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
        parser = argparse.ArgumentParser(description='Plot examples of negation experiment.')
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
        y = item["median"][1]
        points.append((x, y))

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
        sns.scatterplot(x=x_jitter, y=y_jitter, ax=ax)
        sns.kdeplot(x=x_jitter, y=y_jitter, ax=ax, levels=5, fill=True, alpha=0.6, cut=2)

    ax.set_xlabel('P(question)')
    ax.set_ylabel('P(negated)')
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.plot([1, 0], [0, 1], linestyle='--', color='gray')
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
