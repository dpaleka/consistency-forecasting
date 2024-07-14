
# Analysis of Resolved and Upcoming Questions: Methodology and Implications

In this project, we evaluate the accuracy of our forecasting models using a combination of recently resolved questions and those anticipating resolution in the near future.

## Question Categories

### Already Resolved
- [Sample Data](scripts/pipeline/metaculus/questions_cleaned_formatted_20240301_20240601.jsonl)

### Resolves Soon
- [Sample Data](scripts/pipeline/metacluls/questions_cleaned_formatted_20240701_20241001.jsonl)

## Temporal Considerations

Questions that have already been resolved possess a resolution date within the past three months. Conversely, questions expected to resolve soon have a closing date within the upcoming three months. In our case, we use data from Mar 1 - Jun 1, 2024 for questions that already resolved and Jul 1 - Oct 1, 2024 for instances with resolutions in the futures.

### Testing Methodology

- **Past Questions**: Ground truth data is already available from our sources.
- **Future Questions**: We await their expiration, after which human judges manually resolve them. This introduces a bias where the forecast horizon influences the certainty and accuracy of predictions. Specifically, questions resolving soon have a limited timeframe, leading to more precise forecasts, whereas questions already resolved could theoretically have a broader resolution window, adding complexity.

### Illustrative Examples

1. **Questions with Retrospective Resolution**:
   - A question initially set to close far in the future may resolve unexpectedly early, impacting its inclusion in the resolved question set.
     - *Example*: "Will Alexei Navalny become the president of Russia by 2050?" If evaluated retrospectively on April 1, 2024 this question would resolve early as Navalny died on February 16 and would be included in the set.

2. **Questions with Imminent Resolution**:
   - These questions must close within a fixed three-month period, simplifying the forecasting challenge.
     - *Example*: "Will Joe Biden win the 2024 presidential election?" If evaluated in September 1 (forward-looking), and resolving in November, this question will be present in both prospective and retrospective analyses. However, premature resolution, such as Biden's hypothetical passing in August, would exclude it from both sets.

## Bias Analysis

Forward-looking forecasts involve fixed uncertainty within the defined timeframe, whereas retrospective evaluations must consider a wider range of potential early resolutions. Foreknowledge of the nature of the backward set can, paradoxically, simplify forecasts through conditional probability adjustments. For example, knowing a question about Navalny's presidency resolves between January 1 and April 1, 2024, increases the likelihood of a "no" resolution if Navalny did not ascend to the presidency within this narrow window.

## Corollary on Question Quantity for Each Group

A corollary of this is that, on average, the dataset of questions that are already resolved (looking backwards) will produce more instances than the set of questions that will resolve soon (looking forwards), assuming we are using the same slice of time. The reason being, as the example below will demonstrate, is that the backward-looking set must include all the questions of the forward-looking set of the same timeframe but also additional types of questions as well.

### Example Analysis Using Timeframes

#### Timeframe: 9-12

Questions are analyzed by their creation, resolution, and initially intended closing times across defined periods.

```
Timeframe: 9-12

Example Questions:
A: created on 5, resolves 7, closes 7
A2: created on 5, resolves 7, closes 11
A3: created on 5, resolves 7, closes 15

B: created on 5, resolves 11, closes 11
B2: created on 5, resolves 11, closes 15

C: created on 5, resolves 15, closes 15

D: created on 10, resolves 11, closes 11
D2: created on 10, resolves 11, closes 15

E: created on 10, resolves 15, closes 15

F: created on 15, resolves 16, closes 16
```

### Inclusion Criteria

1. **Resolve Soon (Forward-Looking)**:
   - Evaluated at `cur_time_0 = 8`
   - Includes questions with:
     - `created` before `cur_time_0`
     - `resolves` within the timeframe
     - `closes` within the timeframe
   - Example: Includes `B`

2. **Already Resolved (Backward-Looking)**:
   - Evaluated at `cur_time_1 = 13`
   - Includes questions with:
     - `created` before `cur_time_1`
     - `resolves` within the timeframe
   - Example: Includes `B, B2, D, D2`

## Conclusion

Retrospective datasets typically encompass a broader range of questions compared to prospective datasets, yielding more comprehensive data points for evaluation. This expanded scope enriches the forecaster assessment landscape but introduces complexities related to varied resolution times. Understanding and managing these temporal biases is crucial for enhancing the robustness of our forecasting models.