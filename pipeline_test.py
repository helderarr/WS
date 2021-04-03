import pandas as pd

from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from interfaces import Pipeline

data = pd.read_csv("data/dataset.csv",
                 names=["conversation_utterance_id", "conversation_id", "utterance_id", "rank", "score", "passage"])


pipeline = Pipeline()
pipeline.add_step(DbPediaSpotlightEntityExtractor())
#pipeline.add_step(PageRankPassageRanker())
#pipeline.add_step(TopNPassageSelector())
#pipeline.add_step(BartPassageSumarizer())

pipeline.set_input(data)
pipeline.run()
print(pipeline.get_output())
