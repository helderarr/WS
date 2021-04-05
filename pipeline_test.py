from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PageRankPassageRanker import PageRankPassageRanker
from PassageReader import PassageReader
from interfaces import Pipeline

passage_reader = PassageReader()

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(DbPediaSpotlightEntityExtractor())
pipeline_dbpedia.add_step(PageRankPassageRanker())

pipeline_blink = Pipeline()
pipeline_blink.add_step(DbPediaSpotlightAnnotator())
pipeline_blink.add_step(BlinkEntityExtractor())
pipeline_blink.add_step(PageRankPassageRanker())


utterances = passage_reader.data["conversation_utterance_id"]
utterances = utterances.drop_duplicates()


for utt in list(utterances[1:-1]):

    passages = passage_reader.get_utterance_passages(utt)

    out1 = pipeline_dbpedia.run(passages)
    print(out1)

    out2 = pipeline_blink.run(passages)
    print(out2)
