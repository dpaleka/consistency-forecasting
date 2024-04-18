"""
Write a plotting script that displays a few examples of non-monotonic predictions for supposedly monotonic dta.

Input: a json file that is a list of dicts. Each dict is of the form:
{
        "questions": [
            "What will be the women's 800 meter run world record by the year 2025?",
            “... 2028?",
            "... 2032?",
            “...2036?",
            “... 2040?"
        ],
      “responses”: [
   (irrelevant here)
      ],
     “answers”: [
            [
                {
                    "parsed": true,
                    "result": 113.0
                },
                {
                    "parsed": true,
                    "result": 112.53
                },
                {
                    "parsed": true,
                    "result": 112.78
                },
                {
                    "parsed": true,
                    "result": 112.28
                },
                {
                    "parsed": true,
                    "result": 112.78
                },
                {
                    "parsed": true,
                    "result": 112.78
                }
            ],
          (… answers to other questions)
      ],
        "type": "monotonic_sequence",
        "direction": "decreasing",
        "median": [
            112.78,
            112.28,
            112.78,
            112.655,
            112.53
        ],
        "std_devs": [
            0.2287951534063213,
            0.5,
            1.4210854715202004e-14,
            0.5137011669140802,
            0.3660790746394675
        ]
},

Plot several of these on a graph with x-axis equally spaced [2025, 2028, 2032, 2036, 2040]. Connect the 5 points for the same question tuple with a broken line of a distinctive color. The list of all question tuples you should plot is a global variable, storing the first question in each tuple: QUESTIONS_TO_PLOT=[""What will be the women's 800 meter run world record by the year 2025?”, …] 

The medians are all on different scales; normalize to the mean of the medians so that all tuples you plot are on roughly the same scale.

"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from plot_utils import get_dimensions, colors_matplotlib

LABELS : dict[str, str] = {
    "What will be the women's 800 meter run world record by the year 2025?" : 'Women 800m world record',
    "How many people will have completed a solo, unassisted trek to both the North and South Poles by the year 2025?" : 'Solo treks to North and South Pole',
    "What will be the men's javelin throw world record by the year 2025?" : 'Men javelin world record',
    "How many new medicines will be approved by the FDA until the year 2025?" : 'New medicines approved by FDA',
}


QUESTIONS_TO_PLOT = LABELS.keys()


def get_args():
    def get_parser():
        parser = argparse.ArgumentParser(description='Plot examples of non-monotonicity.')
        parser.add_argument('-i', '--input', type=str, help='The JSON file to read.')
        return parser

    args = get_parser().parse_args()
    
    return args

def normalize_median(medians, normalization='mean_std') -> tuple[list[float], dict]:
    if normalization == 'mean_std':
        mean_median = np.mean(medians)
        std_median = np.std(medians)
        return [(median - mean_median) / std_median for median in medians], {'mean': mean_median, 'std': std_median}
    elif normalization == 'min_max':
        min_median = np.min(medians)
        max_median = np.max(medians)
        return [(median - min_median) / (max_median - min_median) for median in medians], None
    # The following two look quite bad. We would need breaks in the y-axis to make them look good.
    elif normalization == "log":
        return [np.log(median) for median in medians], None
    elif normalization == "none":
        return medians, None


def plot_non_monotonic(input_file, error_bars=False):
    with open(input_file, 'r') as f:
        data = json.load(f)

    x = [2025, 2028, 2032, 2036, 2040]

    #colors = plt.cm.get_cmap('tab10', len(QUESTIONS_TO_PLOT))
    colors = plt.cm.get_cmap('tab10', 4)

    # circle, square, triangle, diamond
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    normalization='mean_std'
    dimension_factor = 2
    dim_0, dim_1 = get_dimensions()
    plt.figure(figsize=(dimension_factor * dim_0, dimension_factor * dim_1))


    for idx, question in enumerate(QUESTIONS_TO_PLOT):
        for entry in data:
            if entry['questions'][0] == question:
                print()
                normalized_medians, norm_params = normalize_median(entry['median'], normalization=normalization)
                # use different lines for different questions
                #plt.plot(x, normalized_medians, marker='o', linestyle='--', color=colors(idx), label=LABELS[question])

                if error_bars:
                    # entry['answers'] is a list of lists of dicts, {parsed: bool, result: float | null}
                    # get the max and min
                    # not sure what this does if an answer is None or sth
                    #import code; code.interact(local=dict(globals(), **locals()))
                    robust_max_answers, robust_min_answers = [], []
                    all_clean_answers = []
                    std_devs = []
                    for answers in entry['answers']:
                        #print(f"\n{answers}\n")
                        clean_answers = [answer['result'] for answer in answers if answer['parsed']]
                        offset = 1
                        assert len(clean_answers) >= 2 * offset + 1, 'cannot compute robust max/min with less than 3 answers'
                        clean_answers.sort()
                        std_devs.append(np.std(clean_answers))
                        robust_max_answers.append(clean_answers[-1 - offset])
                        robust_min_answers.append(clean_answers[0 + offset])
                        all_clean_answers.append(clean_answers)


                    if idx == 3:
                        print("FDA question")
                        print(all_clean_answers)

                    print(f"Norm params: {norm_params}")

                    assert normalization=='mean_std', 'Error bars only implemented for mean_std normalization'
                    # normalize using norm params
                    robust_max_answers = [(answer - norm_params['mean']) / norm_params['std'] for answer in robust_max_answers]
                    robust_min_answers = [(answer - norm_params['mean']) / norm_params['std'] for answer in robust_min_answers]
                    std_devs = [(std_dev) / norm_params['std'] for std_dev in std_devs]
                    all_clean_answers = [[(answer - norm_params['mean']) / norm_params['std'] for answer in clean_answers] for clean_answers in all_clean_answers]  

                    #vertical_offset = 5 * (len(QUESTIONS_TO_PLOT) - idx - 1)
                    vertical_offset = 5 * idx

                    visual_factor = 1
                    robust_max_answers = [robust_max_answer * visual_factor + vertical_offset for robust_max_answer in robust_max_answers]
                    robust_min_answers = [robust_min_answer * visual_factor + vertical_offset for robust_min_answer in robust_min_answers]
                    all_clean_answers = [[answer * visual_factor + vertical_offset for answer in clean_answers] for clean_answers in all_clean_answers]
                    normalized_medians = [normalized_median * visual_factor + vertical_offset for normalized_median in normalized_medians]


                    # make box plot
                    print(x)
                    # don't show x ticks
                    plt.boxplot(all_clean_answers, positions=x, widths=0.2, showfliers=False, showmeans=True, meanline=True, meanprops={'color': colors(idx)})


                    # put this thing in the background
                    plt.plot(x, normalized_medians, marker=markers[idx], linestyle=linestyles[idx], color=colors(idx), label=LABELS[question], zorder=0)

                    # plot, only two dashes on that x-coordinate on the corresponding y-coordinate, for both min and max, with that color
                    #plt.plot(x, robust_max_answers, marker='_', linestyle='None', color=colors(idx))
                    #plt.plot(x, robust_min_answers, marker='_', linestyle='None', color=colors(idx))
                    # color the line between the two marks
                    # for i in range(len(x)):
                    #     plt.plot([x[i], x[i]], [robust_min_answers[i], robust_max_answers[i]], marker='_', linestyle='-', color=colors(idx))

                    # use fill_between
                    #normalized_medians = np.array(normalized_medians)
                    #std_devs = np.array(std_devs)
                    #plt.fill_between(x, normalized_medians + std_devs, normalized_medians - std_devs, alpha=0.2, color=colors(idx))
                    





    #plt.xlabel('Years')
    # remove the x label
    #plt.xlabel('')
    plt.xticks([2025, 2030, 2035, 2040], ['2025', '2030', '2035', '2040'], fontsize=12)

    # please use those x-ticks
    #plt.tick_params(axis='x', which='major', labelsize=12)


    # larger font
    plt.ylabel('Normalized prediction', fontsize=16)
    # make legend font larger

    # no ticks on y axis
    plt.yticks([])

    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    # make legend below
    plt.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc='upper center', ncol=1, fontsize='12')
    

    input_file = Path(input_file)
    Path(f'new_plots/').mkdir(parents=True, exist_ok=True)
    filename_stem = f'new_plots/{input_file.stem}_questions_{len(QUESTIONS_TO_PLOT)}'
    if error_bars:
        filename_stem += 'error_bars'

    plt.savefig(f'{filename_stem}.png', bbox_inches='tight')
    plt.savefig(f'{filename_stem}.pdf', bbox_inches='tight')

    plt.grid()
    plt.show()

if __name__ == '__main__':
    args = get_args()
    plot_non_monotonic(args.input, error_bars=True)