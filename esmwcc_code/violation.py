#%%
import numpy as np
import scipy.stats as stats
from utils import TYPES, SCALES

#%%
def spearmanr_subset(results, indices_to_skip : list, direction : str):
    assert direction in ['increasing', 'decreasing']
    if len(indices_to_skip) > 0:
        assert max(indices_to_skip) < len(results) and min(indices_to_skip) >= 0

    subset_results = [results[i] for i in range(len(results)) if i not in indices_to_skip]
    if len(subset_results) < 2:
        return {'correlation': 1.0, 'p_value': 0.0}

    if direction == 'increasing':
        correlation, p_value = stats.spearmanr(subset_results, [i for i in range(len(subset_results))])
    elif direction == 'decreasing':
        print("subset_results", subset_results)
        print("reversed(range(len(subset_results)))", [i for i in reversed(range(len(subset_results)))])
        correlation, p_value = stats.spearmanr(subset_results, [i for i in reversed(range(len(subset_results)))])
    
    return {'correlation': correlation, 'p_value': p_value}


#%%
def get_violation(entries : list, answer_tuples : list, to_skip : list, type : str, scale : str ='linear'):
    """
        :param entries: list of lists of entries. 
        :param answer_tuples: list of dicts of tuples
        :param to_skip: list of sets of indices to skip (because they are not parsed or the answer is invalid). currently only used for compas tasks
        :param type: the type of the property to check
        :param scale: 'linear' or 'log'
    """

    assert len(entries) == len(answer_tuples)
    assert type in TYPES and scale in SCALES
    num_tasks = len(entries)

    violations = []

    for entry, answer_tuple, indices_to_skip in zip(entries, answer_tuples, to_skip):
        task = entry['questions'] # lists of strings. serves mainly for debugging, or getting outliers
        print("indices_to_skip", indices_to_skip)
        assert len(task) == len(answer_tuple['means'])
        num_questions = len(task)

        if type == 'negated_pair':
            assert num_questions == 2
            if scale == 'linear':
                #violation = (sum(answer_tuple['means']) - 1)
                violation = abs(sum(answer_tuple['means']) - 1)
                # not sure what to do with the standard deviations here
            elif scale == 'log':
                violation = (sum([np.log(p/(1-p)) for p in answer_tuple['means']]) - 0)
                # not sure what to do with the standard deviations here
        
        elif type == 'bayes':
            assert num_questions == 4
            if scale == 'linear':
                #violation = (answer_tuple['means'][0] * answer_tuple['means'][2] - answer_tuple['means'][1] * answer_tuple['means'][3])
                violation = np.sqrt(abs(answer_tuple['means'][0] * answer_tuple['means'][2] - answer_tuple['means'][1] * answer_tuple['means'][3]))
                # not sure what to do with the standard deviations here
            elif scale == 'log':
                violation = (np.log(answer_tuple['means'][0]) + np.log(answer_tuple['means'][2]) - np.log(answer_tuple['means'][1]) - np.log(answer_tuple['means'][3]))
                # not sure what to do with the standard deviations here
        
        elif type == 'precursor_event':
            # The second question is the precursor event, should have larger probability
            assert num_questions == 2
            if scale == 'linear':
                violation = max(0, answer_tuple['means'][0] - answer_tuple['means'][1])
                # not sure what to do with the standard deviations here
            elif scale == 'log':
                violation = max(0, np.log(answer_tuple['means'][0]) - np.log(answer_tuple['means'][1]))
                # not sure what to do with the standard deviations here
        
        elif type == 'compas_bail':
            # Q: Do you this person should be released on bail? (1 = yes, 0 = no)
            # weaker indices have committed less crimes than the anchor, thus should have larger probability of being released
            # stronger indices have committed more crimes than the anchor, thus should have smaller probability of being released
            if scale == 'linear':
                weaker_indices = [i for i in range(1, 6) if i not in indices_to_skip]
                stronger_indices = [i for i in range(6, 11) if i not in indices_to_skip]

                if 0 in indices_to_skip:
                    # We couldn't parse the anchor, so there is no violation to be calculated. 
                    violation = 0.0

                else:
                    anchor_val = answer_tuple['means'][0]

                    violation = np.sum(np.maximum(0, anchor_val - np.array(answer_tuple['means'])[weaker_indices])) \
                                + np.sum(np.maximum(0, np.array(answer_tuple['means'])[stronger_indices] - anchor_val))
                
                    #violation /= len(weaker_indices) + len(stronger_indices)

            elif scale == 'log':
                assert False, f"Cannot plot log scale for {type}"
        
        elif type == 'monotonic_sequence':
            # Use Spearman's rank correlation coefficient
            # The inputs should be numbers on an arbitrary scale
            
            if scale == 'linear':
                # skip the indices to skip
                spearmanr_data : dict = spearmanr_subset(answer_tuple['means'], indices_to_skip=indices_to_skip, direction=entry['direction'])
                spearmanr_statistic, p_value = spearmanr_data['correlation'], spearmanr_data['p_value']
                violation = (1 - spearmanr_statistic) / 2  # because spearmanr_statistic is between -1 and 1, with 1 for perfect order

            elif scale == 'log':
                assert False, f"Cannot plot log scale for {type}"
        
        elif type == 'paraphrase':
            # skip the indices to skip
            if scale == 'linear':
                good_indices = [i for i in range(num_questions) if i not in indices_to_skip]
                if len(good_indices) > 0:
                    max_val = max(answer_tuple['means'][i] for i in good_indices)
                    min_val = min(answer_tuple['means'][i] for i in good_indices)
                    violation = max_val - min_val
                else:
                    #violation = -2
                    violation = 0.0

            elif scale == 'log':
                assert False, f"Cannot plot log scale for {type}"


        
        #for i in range(num_questions):
        #    print(f"Questions: {task[i]}, answer: {answer_tuple['means'][i]}")
        # just for the first
        print(f"First question: {task[0]}, answer: {answer_tuple['means'][0]:.3f}")
        print(f"Violation: {violation:.3f}\n")

        violations.append(violation)
    
    assert len(violations) == num_tasks
    return violations


#%%
def get_normalization_factor(type: str, scale: str = 'linear'):
    # We compare with the baseline of the probability being an uniform distribution on [0,1]
    # Instead of calculating the violation for a random sample, we use the get_violation function
    assert type in TYPES and scale in SCALES

    num_tasks = 2000
    violations = []
    tasks = []
    answer_tuples = []
    # Generate random tasks
    if type == 'negated_pair':
        num_questions = 2
        for i in range(num_tasks):
            tasks.append([f"Question {i}" for i in range(num_questions)])
            answer_tuples.append({'means': np.random.uniform(size=num_questions), \
                                'std_devs': np.zeros(num_questions)})
    elif type == 'bayes':
        num_questions = 4
        for i in range(num_tasks):
            tasks.append([f"Question {i}" for i in range(num_questions)])
            answer_tuples.append({'means': np.random.uniform(size=num_questions), \
                                'std_devs': np.zeros(num_questions)})

    elif type == 'compas_bail' or type == 'compas_recidivism':
        # Binary answers corresponding to YES/NO
        num_questions = 21
        for i in range(num_tasks):
            tasks.append([f"Question {i}" for i in range(num_questions)])
            # here we want 0 or 1
            answer_tuples.append({'means': np.random.randint(2, size=num_questions), \
                                'std_devs': np.zeros(num_questions)})

    elif type == 'monotonic_sequence':
        # Numerical answers on varying scales.
        # We test only scale-invariant monotonicity, so it's OK to get random uniform values, scale doesn't matter
        num_questions = 4
        for i in range(num_tasks):
            tasks.append([f"Question {i}" for i in range(num_questions)])
            answer_tuples.append({'means': np.random.uniform(size=num_questions), \
                                'std_devs': np.zeros(num_questions)})

    violations = get_violation(tasks, answer_tuples, to_skip=[[] for _ in range(num_tasks)], type=type, scale=scale)
    mean_violation = np.mean(violations)
    mean_abs_violation = np.mean(np.abs(violations))
    std_dev_violation = np.std(violations)

    return mean_violation, mean_abs_violation, std_dev_violation

        
#%%
# Test
def test_get_normalization_factor():
    mean_violation, mean_abs_violation, std_dev_violation = get_normalization_factor('negated_pair', scale='linear')
    print(f"{mean_violation=:.3f}, {mean_abs_violation=:.3f}, {std_dev_violation=:.3f}")
    assert abs(mean_violation - 0.) < 0.05 and abs(std_dev_violation - np.sqrt(1/6) < 0.05)
