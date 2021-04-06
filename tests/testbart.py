
from transformers import pipeline


ARTICLE_TO_SUMMARIZE = "Ousmane Dembele scored a dramatic late winner to give Barcelona victory over Real Valladolid " \
                       "and a huge boost in their La Liga title challenge. Barcelona move up to second, " \
                       "one point behind leaders Atletico Madrid, thanks to a sixth La Liga win in a row. Barca " \
                       "looked as if they were going to be frustrated but the turning point was Oscar Plano's " \
                       "straight red card for a dangerous foul on Dembele. Dembele volleyed home in the last minute " \
                       "from Ronald Araujo's flick-on. Both sides hit the woodwork in the first half - Kenan Kodro's " \
                       "header for Valladolid and Pedri's low drive for Barcelona being tipped on to the post. " \
                       "Barcelona's title hopes are in their hands now after Atletico lost to Sevilla at the weekend. " \
                       "Barca visit Real Madrid in El Clasico this Saturday and host Atletico next month. "
to_tokenize = ARTICLE_TO_SUMMARIZE

#[{'summary_text': ' Barcelona beat Real Valladolid 1-0 thanks to a late winner from Ousmane Dembele . Oscar Plano was sent off for a dangerous foul on Dembeel . Barcelona move up to second, one point behind leaders Atletico Madrid .'}]

# Initialize the HuggingFace summarization pipeline
#summarizer = pipeline("summarization")

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
summarized = summarizer(to_tokenize, min_length=50, max_length=120)

# Print summarized text
print(summarized)