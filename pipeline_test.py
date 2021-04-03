import pandas as pd

from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PassageReader import PassageReader
from interfaces import Pipeline


pipeline = Pipeline()
pipeline.add_step(DbPediaSpotlightEntityExtractor())
#pipeline.add_step(PageRankPassageRanker())
#pipeline.add_step(TopNPassageSelector())
#pipeline.add_step(BartPassageSumarizer())

passage_reader = PassageReader()

pipeline.set_input(passage_reader.get_utterance_passages("31_1"))
pipeline.run()
print(pipeline.get_output())
