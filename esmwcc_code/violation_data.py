TYPES = ['negated_pair', 'bayes', 'paraphrase', 'precursor_event', 'compas_bail', 'monotonic_sequence']

from data_wrangling.top_experiments import TOP_EXPERIMENTS

"""
Each experiment is a json containing a list of dicts. Each dict has a "violation" key, which should be taken in the absolute value.
For each experiment, print:
- the mean violation
- the number of valuations over 0
- the percentage of valuations over 0
- the number of valuations over [0.05, 0.1, 0.25, 0.5]
"""

PATH_TO_EXPERIMENTS = 'top_experiments/sorted_violations/'

import json
import os
import numpy as np

def load_experiment_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def transformation_of_metric(x, experiment_type):
    assert x >= 0.
    x = max(x, 0.)
    if experiment_type in ['negated_pair', 'paraphrase', 'precursor_event', 'compas_bail', 'monotonic_sequence']:
        return x
    elif experiment_type == 'bayes':
        #return np.sqrt(x) -> we fixed that in the original violation.py code
        return x
    else:
        raise NotImplementedError


def compute_experiment_metrics(data, experiment_type) -> dict:
    violations = np.array([abs(d['violation']) for d in data])
    violations = np.array([transformation_of_metric(x, experiment_type) for x in violations])  # quick, hacky way. ideally go and modify plot.py
    print(f"{violations[:8]=}")
    print("len(violations):", len(violations))

    mean_violation = np.mean(violations)
    eps = 1e-3
    num_violations_over_0 = np.sum(violations > eps)
    percentage_violations_over_0 = 100 * num_violations_over_0 / len(violations)
    violations_over_thresholds = {
        0.05: np.sum(violations > (0.05 + eps)),
        0.1: np.sum(violations > (0.1 + eps)),
        0.15: np.sum(violations > (0.15 + eps)),
        0.2: np.sum(violations > (0.2 + eps)),
        0.25: np.sum(violations > (0.25 + eps)),
        0.5: np.sum(violations > (0.5 + eps)),
    }
    percentage_violations_over_thresholds = {k: 100 * v / len(violations) for k, v in violations_over_thresholds.items()}

    return {'mean_violation': mean_violation, 'over_0': num_violations_over_0, 'perc_over_0': percentage_violations_over_0, **percentage_violations_over_thresholds}

def bail_metrics(data) -> dict:

    no_to_yes = 0
    yes_to_no = 0

    for d in data:
        # look at the 'median' key; it's a list of 11 numbers
        # the first is the original, the next 5 are the weak counterfactuals, the last 5 are the strong counterfactuals
        # if the original is zero ('NO') and there is at least one strong counterfactual that is 'YES', then it's a N0 -> YES violation
        # if the original is one ('YES') and there is at least one weak counterfactual that is 'NO', then it's a YES -> NO violation
        # otherwise, it might still be a violation, because there is the 0.5 value (UNDECIDED)
        # return the number of violations by type
        # tolerance eps
        
        original = d['median'][0]
        eps = 1e-4
        if original < eps:
            strong_counterfactuals = d['median'][6:]
            if max(strong_counterfactuals) > 1. - eps:
                no_to_yes += 1

        elif original > 1. - eps:
            weak_counterfactuals = d['median'][1:6]
            if min(weak_counterfactuals) < eps:
                yes_to_no += 1
        
    return {'no_to_yes': no_to_yes, 'yes_to_no': yes_to_no}






def get_metadata(experiment_name):
    model_name = 'gpt-4' if 'gpt-4' in experiment_name else 'gpt-3.5-turbo'
    temperature = '0.0' if 'T_0.0' in experiment_name else '0.5'
    experiment_type = None
    experiment_type = 'negated_pair' if 'negated_pair' in experiment_name else experiment_type
    experiment_type = 'bayes' if 'bayes' in experiment_name else experiment_type
    experiment_type = 'paraphrase' if 'paraphrase' in experiment_name else experiment_type
    experiment_type = 'precursor_event' if 'precursor_event' in experiment_name else experiment_type
    experiment_type = 'compas_bail' if 'bail' in experiment_name else experiment_type
    experiment_type = 'monotonic_sequence' if 'monotonic_sequence' in experiment_name else experiment_type
    assert experiment_type is not None

    return model_name, temperature, experiment_type


def analyze_experiments():
    for experiment in TOP_EXPERIMENTS:
        model_name, temperature, experiment_type = get_metadata(experiment)
        file_path = os.path.join(PATH_TO_EXPERIMENTS, experiment + '.json')
        data = load_experiment_data(file_path)
        metrics = compute_experiment_metrics(data, experiment_type)
        mean_violation = metrics['mean_violation']
        num_violations_over_0 = metrics['over_0']
        percentage_violations_over_0 = metrics['perc_over_0']
        percentage_violations_over_thresholds = {k: v for k, v in metrics.items() if k not in ['mean_violation', 'over_0', 'perc_over_0']}

        
        print(f"Results for {experiment}:")
        print(f"Model: {model_name}, temperature: {temperature}, type: {experiment_type}")
        print(f"Mean violation: {mean_violation}")
        #print(f"Number of violations over 0: {num_violations_over_0}")
        #print(f"Percentage of violations over 0: {percentage_violations_over_0}%")
        if 'bail' in experiment:
            metrics = bail_metrics(data)
            print(f"Number of NO->YES violations: {metrics['no_to_yes']}")
            print(f"Number of YES->NO violations: {metrics['yes_to_no']}")

        for threshold, perc in percentage_violations_over_thresholds.items():
            if threshold != 0.2: continue
            print(f"Percentage of violations over {threshold}: {perc.round(3)}%")
        print()

if __name__ == "__main__":
    analyze_experiments()




    