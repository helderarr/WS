from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PassageReader import PassageReader
from interfaces import Pipeline

passage_reader = PassageReader()

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(DbPediaSpotlightEntityExtractor())

pipeline_blink = Pipeline()
pipeline_blink.add_step(DbPediaSpotlightAnnotator())
pipeline_blink.add_step(BlinkEntityExtractor())

for conv in range(31, 80):
    for utt in range(1,9):

        passages = passage_reader.get_utterance_passages(f"{conv}_{utt}")

        out1 = pipeline_dbpedia.run(passages)
        print(out1)

        out2 = pipeline_blink.run(passages)
        print(out2)
