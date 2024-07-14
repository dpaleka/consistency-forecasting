#!/bin/bash

# Call metaculus.py script
python ../metaculus_multipleChoice.py -num 200
python ../count_entries.py -f metaculus_MC.json
python3 ../add_body.py metaculus_MC.json
python3 ../reshape_questions.py --filename metaculus_MC.json
