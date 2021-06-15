from BlinkEntityExtractor2 import BlinkEntityExtractor2
from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PageRankPassageRanker import PageRankPassageRanker
from PageRankPassageRankerMultiLinks import PageRankPassageRankerMultiLinks
from PassageReader import PassageReader
#from GraphPrint import GraphPrint
from SumarizerStep import SumarizerStep
from TopNPassages import TopNPassages
from interfaces import Pipeline
import pandas as pd
from os import path,remove

pd.set_option('mode.chained_assignment', None)


passage_reader = PassageReader()

file_name = "graph.xlsx"

pipeline_blink = Pipeline()
pipeline_blink.add_step(DbPediaSpotlightAnnotator())
pipeline_blink.add_step(BlinkEntityExtractor2())
pipeline_blink.add_step(PageRankPassageRankerMultiLinks(file_name))
#pipeline_blink.add_step(GraphPrint())

#pipeline_blink.add_step(PageRankPassageRanker())
#pipeline_blink.add_step(TopNPassages(3))
#pipeline_blink.add_step(SumarizerStep(100, 200))

utterances = passage_reader.data["conversation_utterance_id"]
utterances = utterances.drop_duplicates()

if path.exists(file_name):
    remove(file_name)

for utt in list(utterances[1:-1]):

    passages = passage_reader.get_utterance_passages(utt)

    out1 = pipeline_blink.run(passages)
    print(out1)

