## Question Generation

The purpose of the script "generate_topic_questions.py" is to generate a set of forecasting questions, in the style of metaculus or manifold, that are:

 - Good Quality: The question is neither obvious nor absurd, and is about an event in the future. To achieve good quality we use 6 examples, 2 of them are always chosen from the inital pool of human curated questions, so that the generations dont stray too much from them. The other 4 are chosen from previously generated questions.
 - Diverse: We dont want to generate questions that ask the same thing, or are too similar between them. To achieve this we use categories, tags, and choose a different set of 6 examples for each generation.
   -Categories: A question can belong to only one category, and it should strictly fit into it. Some categories are: Geopolitics, Space, Natural Sciences ...
   -Tags: A question can have zero, one or more tags, and should be related to them as long as it makes sense inside its category. There are many tags, and they are more specific that categories. Some of them are: Spain, Elon Musk, Tennis, Black holes ...

In each generation we select 1 category, 3 tags and 6 example questions (2 from the initial pool, 2 already generated with the same category, 2 already generated with a different category), and ask the api to generate 3 questions in the category following one or none of the tags. 
