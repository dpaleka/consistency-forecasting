I have generated the test files in scripts/pipeline/metaculus:

metaculus_20240701_20241001.jsonl
metaculus_20240301_20240601.jsonl


The command that I am running and failing is:

'
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240701_20241001.jsonl -d real -o metaculus_cleaned_formatted_20240701_20241001.jsonl --max_questions 25 --overwrite -M gpt-3.5-turbo
Note that normally --max_questions should be 500 and -M should be GPT-4o
'


With "hanging" on this:


```
LOCAL_CACHE: None

 NO_CACHE 


 NO_CACHE 


 NO_CACHE 


 NO_CACHE 

Running <function fetch_question_details_metaculus at 0x10e7a9b20> on 40 datapoints with 10 concurrent queries
<class '__main__.Options'> ['__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'filename']
(venv) ashen@Adams-MacBook-Pro-3 metaculus % 
(venv) ashen@Adams-MacBook-Pro-3 metaculus % 
(venv) ashen@Adams-MacBook-Pro-3 metaculus % 
(venv) ashen@Adams-MacBook-Pro-3 metaculus % 
(venv) ashen@Adams-MacBook-Pro-3 metaculus % 
(venv) ashen@Adams-MacBook-Pro-3 metaculus % ls
add_body_only.sh                                        metaculus_20240701_20241001.jsonl                       scrape_question_RESOLVED.sh
just_format.sh                                          metaculus_MC.json                                       scrape_question_SOON_RESOLVED.sh
metaculus.json                                          metaculus_MC.jsonl                                      scrape_question_mc.sh
metaculus.jsonl                                         metaculus_cleaned_formatted.jsonl                       scrape_questions.sh
metaculus_20240301_20240601.json                        metaculus_cleaned_formatted_20240301_20240601.jsonl     scrape_resolutions_ALL.sh
metaculus_20240301_20240601.jsonl                       metaculus_cleaned_formatted_20240701_20241001.jsonl     test_metaculus.jsonl
metaculus_20240701_20241001.json                        page_content.html
(venv) ashen@Adams-MacBook-Pro-3 metaculus % sh scrape_question_RESOLVED.sh 
total entries: 111
Total entries in the JSON file: 111
LOCAL_CACHE: None

 NO_CACHE 


 NO_CACHE 


 NO_CACHE 


 NO_CACHE 

Running <function fetch_question_details_metaculus at 0x1031adb20> on 111 datapoints with 10 concurrent queries
<class '__main__.Options'> ['__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'filename']
LOCAL_CACHE: None

 NO_CACHE 


 NO_CACHE 


 NO_CACHE 


 NO_CACHE 

The setting for WRITE_VERIFICATION in question_formatter.py: True
The file /Users/ashen/Desktop/consistency-forecasting/src/data/fq/real/metaculus_cleaned_formatted_20240301_20240601.jsonl already exists. Overwriting as per the --overwrite flag.
Running functools.partial(<function validate_and_format_question at 0x1399e6520>, model='gpt-3.5-turbo', fill_in_body=False) on 25 datapoints with 15 concurrent queries
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 314
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 308
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 314
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 322
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 311
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 310
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 337
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 314
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 315
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 314

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 3, 31, 0, 0))
question.body: This question resolves as **Yes** if Bing's market share for the month of March 2024 is 5% or higher [according to Statcounter](https://gs.statcounter.com/search-engine-market-share), using the first value accessed by Metaculus on or after April 15, 2024.
current_date: 2024-07-04 18:28:18.733581
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: The question will resolve **Yes** if the Sacramento Kings reach the Conference Semifinals of the NBA playoffs for the 2023-2024 NBA season.
current_date: 2024-07-04 18:28:18.736394
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 3, 14, 0, 0))
question.body: This question will resolve **Yes** if a nuclear weapon is detonated in or above the territory of a European Union (EU) member state before the dates indicated.  A [deliberate, inadvertent, or accidental/unauthorised](http://scienceandglobalsecurity.org/archive/sgs21barrett.pdf) detonation would result in the question resolving as **Yes**.
current_date: 2024-07-04 18:28:18.870058
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: The question will resolve based on the list of 8 players participating in the 2024 Candidates tournament, according to [FIDE.com](https://wcc.fide.com) and on the official (yet to be launched) tournament website at the start of the 1st game of the tournament.
current_date: 2024-07-04 18:28:18.913433
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: This question resolves as **Yes** if the DPP candidate is inaugurated on 20 May 2024 (or, in the case of a delayed inauguration, is announced to have won by [BBC](https://www.bbc.com/news/world-asia-51077553) and [New York Times](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiRva34lOn8AhWYM1kFHXzCDacQFnoECBAQAQ&url=https%3A%2F%2Fwww.nytimes.com%2F2020%2F01%2F11%2Fworld%2Fasia%2Ftaiwan-election-china.html&usg=AOvVaw0DGpxtWlE6BHtzJzB97RTp)). Otherwise, including the event of a premature Chinese invasion and a toppling of the government (making an election impossible) it resolves as **No**
current_date: 2024-07-04 18:28:18.950327
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve to ”Yes” if, before the listed date, a bill has been signed into law by the President which extends key provisions of the Pandemic and All Hazards Preparedness Act (PAHPA) to at least fiscal year 2027. Resolution will be determined according to reporting from credible sources or from the text of the bill as needed.
current_date: 2024-07-04 18:28:19.020128
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: The question will resolve based on the list of 8 players participating in the 2024 Candidates tournament, according to [FIDE.com](https://wcc.fide.com) and on the official (yet to be launched) tournament website at the start of the 1st game of the tournament.
current_date: 2024-07-04 18:28:19.338992
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** if there is a referendum held in Scotland regarding Scotland's independence from the United Kingdom, and this referendum is held before May 3, 2024.  This referendum may be an "advisory" referendum; that is, it is not necessary for the referendum to have any legally binding effect.

It shall not be deemed resolved by a “wildcat” or “Catalan” style of referendum where the UK government has declined permission for the vote, nor by a vote organised by civil society institutions. A referendum must be deemed to have the consent of the London government
current_date: 2024-07-04 18:28:19.371623
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: The question will resolve based on the list of 8 players participating in the 2024 Candidates tournament, according to [FIDE.com](https://wcc.fide.com) and on the official (yet to be launched) tournament website at the start of the 1st game of the tournament.
current_date: 2024-07-04 18:28:19.843754
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 5, 31, 0, 0))
question.body: The question resolves **Yes** if Erling Haaland is the player with the largest number of goals after each of the 20 PL teams plays their 38th game of the 2023/24 season, [according to Oracle Cloud via the PL website](https://www.premierleague.com/stats/top/players/goals?se=578)
current_date: 2024-07-04 18:28:19.896831
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call
```

Note that adding a -c 1 flag (which forces it to run only 1 coroutine at a time) also doesn't seem to fix it.

'
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240701_20241001.jsonl -d real -o metaculus_cleaned_formatted_20240701_20241001.jsonl --max_questions 25 --overwrite -M gpt-3.5-turbo
Note that normally --max_questions should be 500 and -M should be GPT-4o
'

```
LOCAL_CACHE: None

 NO_CACHE 


 NO_CACHE 


 NO_CACHE 


 NO_CACHE 

The setting for WRITE_VERIFICATION in question_formatter.py: True
The file /Users/ashen/Desktop/consistency-forecasting/src/data/fq/real/metaculus_cleaned_formatted_20240701_20241001.jsonl already exists. Overwriting as per the --overwrite flag.
Running functools.partial(<function validate_and_format_question at 0x12fd5a520>, model='gpt-3.5-turbo', fill_in_body=False) on 25 datapoints with 1 concurrent queries
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 316
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 313
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 312
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 313
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 313
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 301
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 303
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 313
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 315
No date, getting date from the title with an LLM call
Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website.
{'model': 'gpt-3.5-turbo', 'response_model': <class 'common.datatypes.ResolutionDate'>, 'temperature': 0.5} Approx num tokens: 312

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.074957
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: This question will resolve positively if France place in the top 5 (ties resolve positively) at the Olympics being held in 2024. It will resolve ambiguously if the Olympics do not take place in 2024.

The medal table is calculated by taking the total number of medals won by each participating country and ordering by:

1. Number of Gold Medals
2. (Where 1 is tied) Number of Silver Medals
3. (Where 2 is tied) Number of Bronze Medal
current_date: 2024-07-04 18:38:19.101818
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.130401
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.164502
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.174527
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Democratic National Convention as the nominee for the 2024 US Presidential Election.  All other candidates not selected by the DNC will resolve as **No**, for whatever reason (if they are deceased, choose not to seek nomination, withdraw candidacy, are not selected by delegates, or any other reason).  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Democratic nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Democratic National Convention](https://en.wikipedia.org/wiki/Democratic_National_Convention).
current_date: 2024-07-04 18:38:19.182720
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: Note that GE being criminally convicted of fraud is not the same as having a financial situation that is insolvent.  This question aims at the latter.
current_date: 2024-07-04 18:38:19.209929
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: This question will resolve positively if the United States Olympic Team are the (unique) highest ranked team at the 2024 Paris Olympics. It will resolve ambiguously if the Paris Olympics do not take place before 2027. It will resolve negatively if any team achieves a higher or equal ranking to the US team.

*Related questions*

* [How many medals will Team USA win in Paris 2024?](https://www.metaculus.com/questions/7665/total-medals-won-by-the-usa-at-paris-2024/)
* [Will France come in the Top 5 at Paris 2024?](https://www.metaculus.com/questions/7669/france-home-game-advantage/
current_date: 2024-07-04 18:38:19.245517
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2023, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.254666
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call

question_formatter.from_string: resolution_date=ResolutionDate(resolution_date=datetime.datetime(2024, 12, 31, 0, 0))
question.body: This question will resolve as **Yes** for the candidate below who is selected by the Republican National Convention as the nominee for the 2024 US Presidential Election.  All other candidates will resolve as **No**.  This question is not restricted to the candidates currently below; other options may be added in the future.

For this question, it is not relevant who recieves the Republican nomination on the day of the 2024 US election, it is solely determined by who is selected by the delegates of the [Republican National Convention](https://en.wikipedia.org/wiki/Republican_National_Convention).
current_date: 2024-07-04 18:38:19.404445
kwargs: {'model': 'gpt-3.5-turbo'}
No date, getting date from the title with an LLM call
```


However, if you limit the number of questions to 19, it works and generates: metaculus_cleaned_formatted_20240301_20240601.jsonl in the pipeline folder (one level up)
`
python3 ../../../src/format_and_verify_questions.py -f metaculus_20240701_20241001.jsonl -d real -o metaculus_cleaned_formatted_20240701_20241001.jsonl --max_questions 19 --overwrite -M gpt-3.5-turbo -c 1
`

It looks like it breaks as soon as you put in --max_questions 20.


Note that the same pattern of behavior occurs if you replace -f  metaculus_20240701_20241001.jsonl -o metaculus_cleaned_formatted_20240701_20241001.jsonl


