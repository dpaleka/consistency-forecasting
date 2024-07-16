class SyntheticQuestion {
    title string
    category string
    tags string
    feedback string?
    fixed bool?
}

class QuestionGenerationResponse3 {
    question_1 SyntheticQuestion
    question_2 SyntheticQuestion
    question_3 SyntheticQuestion
}

class GenerateQuestionInput {
    example_questions SyntheticQuestion[]
    category string
    tags string[]
}

function GenerateSyntheticQuestions(input: GenerateQuestionInput) -> QuestionGenerationResponse3 {
    client GPT4o

    prompt #"
        I want you to help me generate three forecasting questions for a forecasting market site like Metaculus or PredictIt.
        I will provide you with a category and some tags. Your task is to generate questions that can be answered with a probability between 0 and 1.
        For each tag, generate a relevant question if the tag is pertinent to the category. If the tag is not relevant, generate a general question about the category.
        Examples:
        {% for question in input.example_questions %}
        Example:
        - Title: {{ question.title }}
        - Category: {{ question.category }}
        - Tags: {{ question.tags | join(', ') }}
        {% endfor %}
        ----
        Category: {{ input.category }}
        Tags: {{ input.tags | join(', ') }}

        {{ ctx.output_format }}
    "#
}

test generate_synthetic_questions_test {
  functions [GenerateSyntheticQuestions]
  args {
    input {
      example_questions [
        { 
          title #"Impact of AI on Global Economy"#
          category #"Economics"#
          tags [#"AI"#, #"global"#, #"economy"#]
        }
        { 
          title #"Probability of Interstellar Travel by 2100"#
          category #"Technology"#
          tags [#"space"#, #"travel"#, #"future"#]
        }
      ]
      category #"Science"#
      tags [#"innovation"#, #"research"#]
    }
  }
}