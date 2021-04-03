from flair.models import MultiTagger

tagger = MultiTagger.load("hunflair")
from flair.data import Sentence

sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome")

sentence = Sentence("Mouth cancer, also known as oral cancer, is where a tumour develops in the lining of the mouth. ")


sentence = Sentence("Throat cancer is a general term that usually refers to cancer of the pharynx and/or larynx. Regions included when considering throat cancer include the nasopharynx, oropharynx, hypopharynx, glottis, supraglottis and subglottis; about half of throat cancers develop in the larynx and the other half in the pharynx.")
# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())