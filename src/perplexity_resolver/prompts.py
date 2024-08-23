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
- answer (Optional): A boolean value, True if the answer to the question is positive, False if it is negative. This field should be empty if can_resolve_question is False. But it never should be empty if can_resolve_question is True.

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

<resolver_output>
  <chain_of_thought>
    Based on current information:
    1. The IOC has set a quota of 10,500 athletes for Paris 2024.
    2. Recent reports confirm organizers are adhering to this quota.
    3. No news of significant changes to this quota has emerged.
    Given these factors, it's highly likely the number of athletes will be close to, but not exceed, 10,000.
    
    Sources: IOC press release and recent sports news articles.
  </chain_of_thought>
  <can_resolve_question>true</can_resolve_question>
  <answer>false</answer>
</resolver_output>
"""

perplexity_resolve_example_2 = """
Forecasting Question:
Will a Formula 1 Grand Prix be hosted in a country currently under international sanctions by 2025?
Details:
This question will resolve as Yes if, by December 31, 2025, a Formula 1 Grand Prix is officially scheduled to take place in a country that is currently under international sanctions imposed by recognized authorities such as the United Nations, the European Union, or the United States. The term 'international sanctions' refers to restrictions imposed on a country by the international community, which may include trade restrictions, financial embargoes, and travel bans. The hosting of the Grand Prix must be confirmed by an official announcement from Formula 1 or the relevant governing body. If the event is canceled or relocated for any reason, this question will resolve as No. Temporary or unofficial races do not count towards this resolution.
<resolver_output>
  <chain_of_thought>
    Based on current information:
    1. The Formula 1 calendar for 2025 has not been finalized yet.
    2. There are currently no official announcements about F1 races in sanctioned countries.
    3. The political landscape and international relations can change rapidly.
    4. F1 has shown willingness to expand into new markets in recent years.
    Even though it is currently unlikely, we cannot yet be certain that it will not happen. The situation could change before the end of 2025, and a race could potentially be scheduled in a sanctioned country. Therefore, the question cannot be resolved at this time.
  </chain_of_thought>
  <can_resolve_question>false</can_resolve_question>
  <answer>null</answer>
</resolver_output>
"""

perplexity_resolve_example_3 = """
Forecasting Question:
Will a commercial space flight carry more than 20 tourists to Low Earth Orbit in a single launch by the end of 2026?

Details:
This question will resolve as Yes if, by December 31, 2026, a single commercial space flight carries more than 20 paying tourists to Low Earth Orbit (LEO). A "tourist" is defined as a paying customer who is not a professional astronaut or employee of the space flight operator. The flight must reach an altitude of at least 160 km (100 miles) to be considered LEO. The total number will be based on official reports from the commercial space flight operator or verified third-party sources. The tourists must all be on the same launch vehicle, and the flight does not need to dock with a space station to count.

<resolver_output>
  <chain_of_thought>
    Current space tourism flights carry 4-6 passengers max. Scaling to 20+ presents significant technical and regulatory hurdles. While rapid progress is being made in commercial spaceflight, achieving this capacity by 2026 is uncertain. No concrete plans for such large-capacity tourist flights have been announced. Given the industry's unpredictable nature and potential for breakthroughs, it's premature to resolve this question definitively.
  </chain_of_thought>
  <can_resolve_question>false</can_resolve_question>
  <answer>null</answer>
</resolver_output>
"""

# Prompt for Perplexity to resolve forecasting questions
perplexity_resolve_prompt = """
You are provided with a forecasting question and additional details.
Based on the available information, determine if the question can be resolved and with what result.
Be conservative, if there is not enough information to make a definitive conclusion, indicate that the question cannot be resolved.

Provide your analysis and answer in the following XML structure:

<resolver_output>
  <chain_of_thought>
    [Provide your step-by-step reasoning here, including analysis of the available information and any relevant sources.]
  </chain_of_thought>
  <can_resolve_question>
    [Insert 'true' if the question can be resolved, 'false' if it cannot]
  </can_resolve_question>
  <answer>
    [If can_resolve_question is true, insert 'true' for an affirmative resolution, 'false' for a negative resolution. If can_resolve_question is false, leave this element empty]
  </answer>
</resolver_output>

Ensure your chain_of_thought provides a comprehensive analysis, and include citations to relevant sources where applicable.

<example>
{example_1}
<example>

<example>
{example_2}
<example>

<example>
{example_3}
<example>

Analyze the following forecasting question and provide your response in XML format. Remember the task is to search for relevant information and determine if the question can be resolved based on the available data.
If there is not enough information or the resolution may change in the future, indicate that the question cannot be resolved.

Forecasting Question:
{question_title}
Details:
{question_body}
"""
