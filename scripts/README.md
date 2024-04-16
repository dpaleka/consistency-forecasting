# Running a scraping session
scrape_questions.sh: runs the entire pipeline

## Scripts inside scrape_questions.sh
metaculus.py: Scrapes 350 questions from metaculus api
reformat_entries.py:  Adds datetime object and ensures only binary questions are kept in question set (narrows down to 175 questions)
add_body.py --filename QUESTIONS_CLEANED_MODIFIED.json:   Selenium webdriver scrapes for description and resolution criteria of json file provided
reshape_metaculus.py --filename QUESTIONS_CLEANED_MODIFIED.json:  Reformats json file into correct instantiation jsonl file

## Other scripts
scraping_sites_easy:  Folder containing code to scrape all other websites
scrape_sel_sample.py: Helper function for add_body.py for selenium webdriver
count_entries.py -f filename: Utility to count number of entries in json file
inspect_jsonl.py: Checks whether jsonl in correct format

## Generated Files
metaculus.json: initial scraped questions from metaculus.py
QUESTIONS_CLEANED.json: copy of metaculus.json
QUESTIONS_CLEANED_MODIFIED.json: Final json that contains the body and resolution criteria of binary questions
QUESTIONS_CLEANED_MODIFIED.jsonl: Final jsonl file in correct format to use for instantiator

## Things most likely safe to ignore
master_questions_list:  Has preliminary scraped questions from predictit, metaculus, and manifold
testing_prob_not_useful:  scripts used to get selenium working
not_useful:  code not being used