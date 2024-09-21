parse_resolver_output_prompt = """
In the following text there is the answer given by the resolver to a forecasting question. The resolver is meant to answer if the question can be resolved.
You must find the appropriate values for the fields: 
- chain_of_thought: A string with the reasoning of the resolver.
- can_resolve_question: A boolean value, True if the question can be resolved, False otherwise.
- answer (Optional): A boolean value, True if the answer to the question is positive, False if it is negative. This field should be empty if can_resolve_question is False. But it never should be empty if can_resolve_question is True.

The answer was meant to be in XML format, it may be bad formated xml, like in this example:

Question:
Will the 2024 Summer Olympics in Paris have over 10,000 athletes participating?

Resolver Output:
<resolver_output>
  <chain_of_thought>
    Based on current information:
    1. The IOC has set a quota of 10,500 athletes for Paris 2024.
    2. Recent reports confirm organizers are adhering to this quota.
    3. No news of significant changes to this quota has emerged.
    Given these factors, it's highly likely the number of athletes will be close to, but not exceed, 10,000.
    
    Sources: IOC press release and recent sports news articles.
  </chain_of_though>
  <can_resolve_question>true</can_resolve_question>
  <answer>false</answer>
</resolver_out>

ResolverOutput(
    chain_of_thought="Based on current information: 1. The IOC has set a quota of 10,500 athletes for Paris 2024. 2. Recent reports confirm organizers are adhering to this quota. 3. No news of significant changes to this quota has emerged. Given these factors, it's highly likely the number of athletes will be close to, but not exceed, 10,000. Sources: IOC press release and recent sports news articles.",
    can_resolve_question=True,
    answer=False,
)

Or not even xml at all, like in this other example:

Question:
Spain will win 5 or more gold medals at the 2024 Paris Olympics.

Resolver Output:

'The forecasting question can be resolved based on the available information. According to the sources, Spain won 5 gold medals at the 2024 Paris Olympics.
The answer to the question is true.'


ResolverOutput(
    chain_of_thought="
    can_resolve_question=True,
    answer=True,
)
-----
In the following text there is the answer given by the resolver to a forecasting question. The resolver is meant to answer if the question can be resolved.
You must find the appropriate values for the fields: 
- chain_of_thought: A string with the reasoning of the resolver.
- can_resolve_question: A boolean value, True if the question can be resolved, False otherwise.
- answer (Optional): A boolean value, True if the answer to the question is positive, False if it is negative. This field should be empty if can_resolve_question is False. If can_resolve_question is True, this field should be filled, either with True or False.

Question:
{question_title}

Resolver Output:
{full_string}
"""


perplexity_resolve_example_1 = """
Forecasting Question:
Will the 2024 Summer Olympics in Paris have over 10,000 athletes participating?

Details:
This question will resolve positively if the official number of participating athletes exceeds 10,000. It will resolve negatively if the number is 10,000 or fewer. The final count will be based on the number of athletes who actually participate in the Games, not just those initially registered.
Resolution Date: 2024-08-05

<resolver_output>
  <chain_of_thought>
    Based on current information:
    1. The IOC has set a quota of 10,500 athletes for Paris 2024.
    2. Recent reports confirm organizers adhered to this quota.
    3. No news of significant changes to this quota has emerged.
    Given these factors, it's highly likely the number of athletes was over 10,000.
    
    Sources: IOC press release and recent sports news articles.
  </chain_of_thought>
  <can_resolve_question>true</can_resolve_question>
  <answer>true</answer>
</resolver_output>
"""

perplexity_resolve_example_2 = """
Forecasting Question:
Will a commercial space flight carry more than 20 tourists to Low Earth Orbit in a single launch by the end of 2026?

Details:
This question will resolve as Yes if, by December 31, 2026, a single commercial space flight carries more than 20 paying tourists to Low Earth Orbit (LEO). A "tourist" is defined as a paying customer who is not a professional astronaut or employee of the space flight operator. The flight must reach an altitude of at least 160 km (100 miles) to be considered LEO. The total number will be based on official reports from the commercial space flight operator or verified third-party sources. The tourists must all be on the same launch vehicle, and the flight does not need to dock with a space station to count.

<resolver_output>
  <chain_of_thought>
    The question is about events up to 2026; we only check whether we can resolve based on current information.
    In 2024, the largest *crew count* for a flight offering any sort of space tourism was 8 in the Blue Origin New Shepard launch on August 29, 2024.
    The flight carried 6 tourists under a broad understanding of the term, and reached about 100km above earth, which means it does not satisfy the question's criteria.
    The Crew Dragon launches up to 2024 did reach low earth orbit, but carried only 4 crew members.

    There is no information about future flights, so we cannot yet be certain, hence the question cannot be resolved.
  </chain_of_thought>
  <can_resolve_question>false</can_resolve_question>
  <answer>null</answer>
</resolver_output>
"""

perplexity_resolve_example_3 = """
Forecasting Question:
Will Spain win 4 or more gold medals in the 2024 Summer Olympics in Paris?
Details:
This question will resolve positively if Spain wins 4 or more gold medals during the 2024 Summer Olympics in Paris. It will resolve negatively if Spain wins 3 or fewer gold medals. The final count will be based on the official medal tally at the conclusion of the Games.
<resolver_output>
<chain_of_thought>
The official Wikipedia page "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table" shows that Spain has won 5 gold medals in the 2024 Summer Olympics in Paris.
Wikipedia is generally trustworthy on these matters, and there is no strong reason to believe this is not the final count, so the question can be resolved YES.
</chain_of_thought>
<can_resolve_question>true</can_resolve_question>
<answer>true</answer>
</resolver_output>
"""

perplexity_resolve_example_5 = """
Forecasting Question:
Will the struck-down part of Texas's 2023 election law be reinstated by July 2024?
Details:
This question resolves as YES if, by July 31, 2024, the part of Texas's 2023 election law that was struck down by a federal judge (specifically, the restrictions on voters with disabilities casting absentee ballots) is reinstated through legal or legislative action. The reinstatement must be confirmed through official legal documents or announcements and reported by at least two reputable news sources. If no such reinstatement occurs by the resolution date, or if the law remains partially or fully struck down, the question resolves as NO. Note that this question pertains only to the specific part of the law related to restrictions on voters with disabilities and absentee ballots, not the entire 2023 election law.
Resolution Date: 2024-07-31
Created Date: 2024-06-30
<resolver_output>
  <chain_of_thought>
    It is not clear that the question is even well-formed. To the best of my knowledge, there is no Texas law that restricts voters with disabilities from casting absentee ballots.
    The question itself would resolve to NO, given given that nothing is being reinstated. 
    But, given that my search did not find anything about this, I decide to say say that the question cannot be resolved.
  </chain_of_thought>
  <can_resolve_question>false</can_resolve_question>
  <answer>null</answer>
</resolver_output>
"""

perplexity_resolve_example_4 = """
Forecasting Question:
Will the South Korean Presidential Office announce conversion of additional vehicles to biofuel by July 2024?

Details:
This question will resolve as YES if, by July 31, 2024, the South Korean Presidential Office officially announces that additional vehicles, beyond those already known, have been converted or are in the process of being converted to run on biofuel. The announcement must come from official channels or be reported by at least two major, reputable news outlets (such as BBC, Reuters, Associated Press, or The Guardian). The additional vehicles must be part of the presidential fleet or directly associated with governmental transportation. If no such announcement is made by the specified date, the question will resolve as NO.

Resolution Date: 2024-07-31
Created Date: 2024-06-01

<resolver_output>
  <chain_of_thought>
    To resolve this question, I've checked the following sources:
    1. Official channels of the South Korean Presidential Office
    2. Major reputable news outlets including BBC, Reuters, Associated Press, and The Guardian
    
    After a thorough search, I found no announcements or reports about the South Korean Presidential Office converting additional vehicles to biofuel. Specifically:
    - The official website and social media accounts of the South Korean Presidential Office do not contain any such announcement.
    - None of the specified major news outlets (BBC, Reuters, AP, The Guardian) have reported on this topic.
    - No other reputable sources have covered this announcement.

    The question clearly states that if no such announcement is made by July 31, 2024, the question will resolve as NO. Given that we have reached this date and no announcement has been made through the specified channels, we can confidently resolve this question.
  </chain_of_thought>
  <can_resolve_question>true</can_resolve_question>
  <answer>false</answer>
</resolver_output>
"""


# Prompt for Perplexity to resolve forecasting questions
perplexity_resolve_prompt = """
You are provided with a forecasting question and additional details.
Based on the available information, determine if the question can be resolved and with what result.
Be conservative, if there is not enough information to make a definitive YES or NO conclusion, indicate that the question cannot be resolved by using the tag <can_resolve_question>false</can_resolve_question>.
If the question is not well formed, also indicate that the question cannot be resolved by using the tag <can_resolve_question>false</can_resolve_question>.

Tip 1: If the question asks whether an event will happen "by a certain date", the "Created Date" field in the question is important. For example, if the question was created on 15 May 2024, and it asks about an event happening by 31 July 2024, then only events between 15 May 2024 and 31 July 2024 are relevant for your answer.

Tip 2: Do not answer solely based on the "Title" field of the question. The precise resolution criteria of the question are in the "Details" field.

Tip 3: The resolution criteria and dates in the "Details" field is binding, and the Resolution Date is only when the question was originally supposed to be resolved YES or NO. For example, if the question is about the Olympics 2024, the "Resolution Date" field might have a date in August 2024 even though the Olympics ended on 30 July 2024; this is just because, perhaps, the details of the question are about counting certain achievements, and the count is done after the Olympics.

Tip 4: If the "Created Date" field is not provided, just ignore it in your reasoning.

Tip 5: If, after reasoning, you are not sure about the answer, just say so and set <can_resolve_question>false</can_resolve_question>. It is much better to err on the side of caution and not provide an answer than to provide an answer that may be incorrect.

Tip 6: Some questions are not well formed. If the question background talks about past events that never actually happened in order to set up a context for a question that is about the future, say that the question cannot be resolved.

Tip 7: However, if you do understand the question, it is well-formed, and it is clear there is no evidence for it to resolve YES, then you should answer NO.


Provide your analysis and answer in the following XML structure:

<resolver_output>
  <chain_of_thought>
    [Provide your step-by-step reasoning here, including analysis of the available information and any relevant sources. Cite the fragments of information that are relevant. Feel free to use longer chains of thought than in the examples.]
  </chain_of_thought>
  <can_resolve_question>
    [Insert 'true' if the question can be resolved, 'false' if it cannot]
  </can_resolve_question>
  <answer>
    [If can_resolve_question is true, insert 'true' for an affirmative resolution, 'false' for a negative resolution. If can_resolve_question is false, leave this element empty]
  </answer>
</resolver_output>

Ensure your chain_of_thought provides a comprehensive analysis, and include citations to relevant sources where applicable.
Think deeply to ensure your reasoning matches the details of the question exactly. If you are not sure, do not risk it and set <can_resolve_question>false</can_resolve_question>.

<example>
{example_1}
<example>

<example>
{example_2}
<example>

<example>
{example_3}
<example>

<example>
{example_4}
<example>

<example>
{example_5}
</example>

Analyze the following forecasting question and provide your response in XML format. Remember the task is to search for relevant information and determine if the question can be resolved based on the available data.
If there is not enough information or the resolution may change in the future, indicate that the question cannot be resolved. If the question can be resolved, your answer should be either true or false.

Forecasting Question:
{question_title}
Created Date: {created_date}
Resolution Date: {resolution_date}
Details:
{question_body}
"""
