from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PageRankPassageRanker import PageRankPassageRanker
from PassageReader import PassageReader
from SumarizerStep import SumarizerStep
from TopNPassages import TopNPassages
from interfaces import Pipeline
import pandas as pd

pd.set_option('mode.chained_assignment', None)


passage_reader = PassageReader()

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(DbPediaSpotlightEntityExtractor())
pipeline_dbpedia.add_step(PageRankPassageRanker())
pipeline_dbpedia.add_step(TopNPassages(3))
pipeline_dbpedia.add_step(SumarizerStep(100, 200))

pipeline_blink = Pipeline()
pipeline_blink.add_step(DbPediaSpotlightAnnotator())
pipeline_blink.add_step(BlinkEntityExtractor())
pipeline_blink.add_step(PageRankPassageRanker())
pipeline_blink.add_step(TopNPassages(3))
pipeline_blink.add_step(SumarizerStep(100, 200))

utterances = passage_reader.data["conversation_utterance_id"]
utterances = utterances.drop_duplicates()

for utt in list(utterances[1:-1]):

    passages = passage_reader.get_utterance_passages(utt)

    out1 = pipeline_dbpedia.run(passages)
    # print(out1)

    out2 = pipeline_blink.run(passages)
    # print(out2)

    if not out2 == out1:
        print("================================")
        print(out1)
        print("================================")
        print(out2)
        print("================================")
