import pandas as pd

from BlinkAnnotator import BlinkAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PassageReader import PassageReader
from interfaces import Pipeline



passage_reader = PassageReader()

pipeline1 = Pipeline()
pipeline1.add_step(DbPediaSpotlightEntityExtractor())
out1 = pipeline1.run(passage_reader.get_utterance_passages("32_1"))
print(out1)

pipeline2 = Pipeline()
pipeline2.add_step(BlinkAnnotator())
pipeline2.add_step(BlinkEntityExtractor())
out2 = pipeline2.run(passage_reader.get_utterance_passages("32_1"))
print(out2)


