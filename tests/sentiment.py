from flair.data import Sentence
from flair.models import TextClassifier

# load tagger
classifier = TextClassifier.load('sentiment')

# predict for example sentence
sentence = Sentence("enormously entertaining for moviegoers of any age .")
classifier.predict(sentence)

# check prediction
print(sentence)