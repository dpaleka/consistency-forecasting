"""
Read a JSON file that is a list of dicts similar to this:
    {
        "questions": [
            "Will the Taliban capture the Presidential Palace in Kabul by 2026-09-11?",
            "Will the Taliban not capture the Presidential Palace in Kabul by 2026-09-11?"
        ],
        "responses": [
            [
                "[Q] Will the Taliban capture the Presidential Palace in Kabul by 2026-09-11?\n\n[Chain of Thought] The situation in Afghanistan ... reasoning .... [Answer] 0.6"
                ... possibly more responses to the same question
            ],
            [
                "[Q] Will the Taliban not capture the Presidential Palace in Kabul by 2026-09-11?\n\n[Chain of Thought] The situation in Afghanistan ... reasoning ... [Answer] 0.6"
                ... same as above
            ]
        ],
        "answers": [
            [
                {
                    "parsed": true,
                    "result": 0.6
                },
                ... possibly more, corresponding to the responses

            ],
            [
                {
                    "parsed": true,
                    "result": 0.6
                },
                ... same as above
            ]
        ],
        "type": "negated_pair"
    },

(And the number of questions can also vary.)

We want to plot the *violation of a given probabilistic property* across the questions. 
Here, for "type": "negated_pair", the property is that the probabilities of the two questions should sum to 1.
So, we want to plot the difference between the sum of the probabilities of the two questions and 1.
In addition, to handle calibration at low or high probabilities, we may want to produce plots relating to log(p/(1-p)) somehow.

The "type" and the "answers" are the most important parts. 
If there are multiple parsed answers to a given response, we want to use the mean and report the average standard deviation across the responses.
"""

#%%
# Imports
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from utils import TYPES, SCALES, MODELS
from plot_utils import get_dimensions, colors_matplotlib

from violation import get_violation, get_normalization_factor

#%%
# Args
def get_args():
    def get_parser():
        parser = argparse.ArgumentParser(description='Plot the violation of a probabilistic property.')
        parser.add_argument('--input', type=str, help='The JSON file to read.')
        parser.add_argument('--scale', type=str, default='linear', choices=SCALES, help='The scale to use for the violation.')
        return parser

    print(f"{'ipykernel' in sys.modules = }")
    args =  get_parser().parse_args(["--input", "clean_outputs/toy_bayes_gpt-3.5-turbo_method_1shot_china_tg_No_T_0.25_mt_500.json"]) \
        if 'ipykernel' in sys.modules \
        else get_parser().parse_args()
    
    return args
    
args = get_args()

#%%
# Load the data
file = Path(args.input)
with open(file, 'r') as f:
    data = json.load(f)

#%%
from typing import Union, Optional
AnsType = Union[float, str]  # Either a probability, or "YES", or "NO", or "UNKNOWN"


def get_answer_tuple(entry):
    # entry is a dict with keys 'questions', 'responses', 'answers', 'type'
    # answers is a list of lists of dicts

    # Get the mean of all answers for a given question
    num_questions = len(entry['questions'])
    answers = entry['answers']
    mean_answers = []
    std_devs = []

    #print(f"{answers=}")
    indices_to_skip = set()
    print("Answers:", answers)
    for i in range(num_questions):
        question_answers = answers[i]
        #print(f"{question_answers=}"")
        # filter out the answers that are not parsed
        question_answers = [answer["result"] for answer in question_answers if answer['parsed']]
        if len(question_answers) == 0:
            if entry['type'] in ['negated_pair', 'bayes', 'precursor_event']:
                print(f"Warning: no parsed answers for question \"{entry['questions'][i]}\". Replacing with default probability 0.5")
                # TODO think if this is the right thing to do or we should skip the entry
                question_answers = [0.5]
            elif entry['type'] in ['compas_bail', 'compas_recidivism']:
                if i == 0:
                    # We have to skip the whole entry
                    indices_to_skip = set(range(num_questions))
                # Otherwise we'll just have to skip this question
                indices_to_skip.add(i)
            elif entry['type'] in ['monotonic_sequence', 'paraphrase']:
                # the entries are symmetric
                indices_to_skip.add(i)
        
        mean_answers.append(np.median(question_answers) if len(question_answers) > 0 else -10000)
        std_devs.append(np.std(question_answers) if len(question_answers) > 0 else -10000)


    return indices_to_skip, {'means': mean_answers, 'std_devs': std_devs, 'type': entry['type']}


    
#%%
answer_tuples = []
to_skip = [] # a list of sets of indices to skip
for entry in data:
    indices_to_skip, parsed_data = get_answer_tuple(entry)
    answer_tuples.append(parsed_data)
    to_skip.append(indices_to_skip)

    print(f"answer_tuples[-1] = ")


#%%
data_type = data[0]['type']
violations = get_violation(data, answer_tuples, to_skip=to_skip, type=data_type, scale=args.scale)

#%%
# save to clean_outputs/with_violations/filename.json
violations_dir = Path("clean_outputs/with_violations")
violations_dir.mkdir(parents=True, exist_ok=True)
with open(f"{violations_dir}/{file.name}", 'w') as f:
    # merge the data and the violations
    for i in range(len(data)):
        data[i]['violation'] = violations[i]
        # also save the mean/median and std_devs
        data[i]['median'] = answer_tuples[i]['means']
        data[i]['std_devs'] = answer_tuples[i]['std_devs']
    json.dump(data, f, indent=4)

    
nice_names : dict[str, str] = {
    'negated_pair': 'Negation',
    'paraphrase': 'Paraphrase',
    'monotonic_sequence': 'Monotonicity',
    'bayes': 'Bayes Rule',
    'compas_bail': 'COMPAS Bail',
}

#%%
def plot_violations(violations, type : str, scale='linear', normalize=False, additional_info=""):
    assert scale in ['linear', 'log']
    fig, ax = plt.subplots()
    fig.set_size_inches(4.2, 4.2)
    # increase font
    plt.rcParams.update({'font.size': 12})
    # also for the ax labels
    ax.xaxis.label.set_size(12)

    title = f'Consistency violation ({nice_names[type]})'

    if normalize:
        mean_violation, mean_abs_violation, std_dev_violation = get_normalization_factor(type, scale)
        violations = [violation / std_dev_violation for violation in violations]
        ax.set_title(title + ', normalized')
    else:
        ax.set_title(title)

    ax.set_xlabel(f'{additional_info}')

    UPPER_BOUND = 0.6
    for i in range(len(violations)):
        if violations[i] > UPPER_BOUND:
            violations[i] = UPPER_BOUND

    ax.set_xlim(0, UPPER_BOUND)
    if UPPER_BOUND == 0.6:
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6+'])


    ax.hist(violations, bins=12)


    # y-ticks should be integers
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # no space
    #ax.set_ylabel('Number of examples')

    plt.show()
    return fig, ax

    
# quick hack to get the model name: it's either gpt-3.5-turbo or gpt-4 for now, find those
additional_info = ""
if "gpt-3.5-turbo" in args.input:
    # use code font
    additional_info += "gpt-3.5-turbo"
elif "gpt-4" in args.input:
    additional_info += "gpt-4"
additional_info += ", "
if "T_0.0" in args.input:
    additional_info += "T = 0"
elif "T_0.5" in args.input:
    additional_info += "T = 0.5"

fig, ax = plot_violations(violations, data_type, scale=args.scale, normalize=False, additional_info=additional_info)

#%%
# Save the figure, the filename should correspond to the input filename
# mkdir plots
Path("plots").mkdir(parents=True, exist_ok=True)
fig.savefig(f"plots/{file.stem}.png", bbox_inches='tight')
fig.savefig(f"plots/pdfs/{file.stem}.pdf", bbox_inches='tight')


# Run with e.g. python plot.py --input FILENAME.json

# %%
