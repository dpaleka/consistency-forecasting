import json
import requests
from bs4 import BeautifulSoup
import argparse

from scrape_sel_sample import fetch_question_details

def update_questions_with_details(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    
    for question in questions:

        resolution, background = fetch_question_details(question['url'])
        
        question['body'] = {
            "resolution_criteria": resolution,
            "background_info": background
        }
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(questions, file, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Update Metaculus questions with resolution criteria and background info.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing the questions')
    
    args = parser.parse_args()
    
    update_questions_with_details(args.file_path)

if __name__ == '__main__':
    main()