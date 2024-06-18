# Notes on Resolved Questions

In this project, we test the accuracy of our forecaster by using both recently resolved questions as well as questions that will resolve in the near future.

For example

- [Already Resolved] (scripts/pipeline/questions_cleaned_formatted_20240301_20240601.jsonl)
scripts/pipeline/questions_cleaned_formatted_20240301_20240601.jsonl


- [Resolves Soon](scripts/pipeline/questions_cleaned_formatted_20240701_20241001.jsonl)
scripts/pipeline/questions_cleaned_formatted_20240701_20241001.jsonl


Questions that already resolved imply a resolution date (in our case 3 months) in the past while those that resolve soon have an closing date in the near future (also 3 months).  To test the former, we already have the ground truths provided from the source we scraped from.  For the latter, we can wait for expiration to arrive and then a human judge can 'manually' resolve the questions.

However, by the mechanics of this, we introduce a bias into these resolutions.  Namely, for questions that will resolve in the near future, their "latest" possible resolution date is within our timeframe (3 months).  In these cases, a forecaster only has the uncertainty of this timeframe and so we should expect it to make more certain and accurate predictions.

Conversely, for questions that have already resolved the "latest" possible resolution date for those could have been much further into the future, but just so happen to have been resolved in that time frame.

For example, assume it is Jan 1, 2024 and consider the question "Will Alexei Navalny become the president of Russia by 2050?"  This question on Jan 1, would be included in neither set of questions since the closing date is too far in the future.  However, in reality, this question would resolve on Feb 16, the day Alexei Navalny died and therefore would be included in the backward looking, already resolved, question set that is produced on Apr 1, 2024.  

Now consider the following question, "Will Joe Biden win the 2024 presidential election?", which closes in November.  We again consider the circumstances of evaluating this question forward looking (in September), or backward looking (in December).  If eveythnig goes "normally", we will get resolution on November, which will cause this question to be included in both sets.  However, suppose for some reason Joe Biden passes away in October.  Even in this case, both sets would include this.  If Joe Biden passes away in August, then neither set would include it.

Therefore we can see that forward looking sets only have to evaluate questions that close in the time-frame.  Meanwhile, backward looking sets have to consider questions that both close in its given time-frame but also other questions that "resolve early".  On one hand this makes the tasks of forward looking sets easier since they only have to account for the fixed uncertaintly of the timeframe provided while the backward looking sets need to consider a wider timeframe.  However, if the forecaster had knowledge that it was evaluating a question from a backward looking set, then it may actually make the forecast easier / more certain based on this conditional probability.  Using the example of Alexei Navalny as president; suppose the forecaster did not know what the resolution was but did know that it resolved some time between Jan 1 and Apr 1, 2024.  In that case, it would be pretty unusual for Navalny to have actually become president, so the forecaster may have more certainty that the ground truth resolved to 'no'.  In fact, we can think of how the forecaster is essentially answering a different question.  The question would no longer be "Will Alexei Navalny become the president of Russia by 2050?", but instaed "Will Alexei Navalny become the president of Russia some time between Jan 1 and Apr1, 2024?"




## Corallary on question quantity for each group

A corallary of this is that on average, the dataset of questions that are already resolved (looking backwards) will produce more instances than the set of questions that will resolve soon (looking forwards) assuming we are using the same slice of time.  The reason being, as the example below will demonstrate, is that the backwards looking set must include all the questions of the forward looking set of the same time-frame but also with additional types of questions as well.


### Example
Here we assume the timeframe we are evaluating is 9-12.  Therefore, we evaluate this timeframe forwards at time=8 and backwards at time=13.  We then produce every possible grouping of questions in regards to before, during, and after this timeframe; when the question is originally created (created), actually resolves (resolves), and closing time at inception (closes).  Note that resolves <= closes, as a question must resolve before or at its closing time.

```
time_frame: 9-12

actual original closing dates and when it actually resolves

A: created on 5, resolves 7, closes 7
A2: created on 5, resolves 7, closes 11
A3: created 5, resolves 7, closes 15

B: created on 5, resolves 11, closes 11
B2: created on 5, resolves 11, closes 15

C: created 5, resolves 15, closes 15

D: created 10, resolves 11, closes 11
D2: created 10, resolves 11, closes 15

E: created 10, resolves 15, closes 15

F: created 15, resolves 16, closes 16



Inclusion criteria for 'resolve soon' (looking into the future): Evaluate at cur_time_0 = 8
- cur_time_0 > created
- cur_time < resolves
- closes is in time_frame
- Includes: B

Inclusion criteria for 'already resolved' (looking into the past): Evaluate at cur_time_1 = 13
- cur_time_1 > created
- resolves is in time_frame
- Includes: B, B2, D, D2

```


```
Put Code here
```

