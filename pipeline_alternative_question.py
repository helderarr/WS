from DbPediaSpotlightAnnotator import DbPediaSpotlightAnnotator
from BlinkEntityExtractor import BlinkEntityExtractor
from DbPediaSpotlightEntityExtractor import DbPediaSpotlightEntityExtractor
from PageRankPassageRanker import PageRankPassageRanker
from PassageReader import PassageReader
from PrintCenterEntity import PrintCenterEntity
from SumarizerStep import SumarizerStep
from TopNExceptPassages import TopNExceptPassages
from TopNExceptPassagesBoosted import TopNExceptPassagesBoosted
from TopNPassages import TopNPassages
from TopNPassagesWithAlternative import TopNPassagesWithAlternative
from interfaces import Pipeline
import pandas as pd


pd.set_option('mode.chained_assignment', None)


passage_reader = PassageReader()

main_n_passages = 3
alternalive_n_passages = 2

dbPediaSpotlightEntityExtractor = DbPediaSpotlightEntityExtractor()
sumarizerStep = SumarizerStep(50, 80)

pipeline_dbpedia = Pipeline()
pipeline_dbpedia.add_step(dbPediaSpotlightEntityExtractor)
pipeline_dbpedia.add_step(PageRankPassageRanker())
pipeline_dbpedia.add_step(PrintCenterEntity())
pipeline_dbpedia.add_step(TopNPassages(main_n_passages))
pipeline_dbpedia.add_step(sumarizerStep)

pipeline_dbpedia_alt = Pipeline()
pipeline_dbpedia_alt.add_step(dbPediaSpotlightEntityExtractor)
pipeline_dbpedia_alt.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt.add_step(TopNExceptPassages(main_n_passages))
pipeline_dbpedia_alt.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt.add_step(PrintCenterEntity())
pipeline_dbpedia_alt.add_step(TopNPassages(alternalive_n_passages))
pipeline_dbpedia_alt.add_step(sumarizerStep)

pipeline_dbpedia_alt_boosted = Pipeline()
pipeline_dbpedia_alt_boosted.add_step(dbPediaSpotlightEntityExtractor)
pipeline_dbpedia_alt_boosted.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt_boosted.add_step(TopNExceptPassagesBoosted(main_n_passages,boost_factor=10,keep_filtered_entities=True))
pipeline_dbpedia_alt_boosted.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt_boosted.add_step(PrintCenterEntity())
pipeline_dbpedia_alt_boosted.add_step(TopNPassages(alternalive_n_passages))
pipeline_dbpedia_alt_boosted.add_step(sumarizerStep)

pipeline_dbpedia_alt_not_boosted_filtered = Pipeline()
pipeline_dbpedia_alt_not_boosted_filtered.add_step(dbPediaSpotlightEntityExtractor)
pipeline_dbpedia_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt_not_boosted_filtered.add_step(TopNExceptPassagesBoosted(main_n_passages,boost_factor=1,keep_filtered_entities=False))
pipeline_dbpedia_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_dbpedia_alt_not_boosted_filtered.add_step(PrintCenterEntity())
pipeline_dbpedia_alt_not_boosted_filtered.add_step(TopNPassages(alternalive_n_passages))
pipeline_dbpedia_alt_not_boosted_filtered.add_step(sumarizerStep)

dbPediaSpotlightAnnotator = DbPediaSpotlightAnnotator()
blinkEntityExtractor = BlinkEntityExtractor()

pipeline_blink = Pipeline()
pipeline_blink.add_step(dbPediaSpotlightAnnotator)
pipeline_blink.add_step(blinkEntityExtractor)
pipeline_blink.add_step(PageRankPassageRanker())
pipeline_blink.add_step(PrintCenterEntity())
pipeline_blink.add_step(TopNPassages(main_n_passages))
pipeline_blink.add_step(sumarizerStep)

pipeline_blink_alt_not_boosted_filtered = Pipeline()
pipeline_blink_alt_not_boosted_filtered.add_step(dbPediaSpotlightAnnotator)
pipeline_blink_alt_not_boosted_filtered.add_step(blinkEntityExtractor)
pipeline_blink_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_blink_alt_not_boosted_filtered.add_step(TopNExceptPassagesBoosted(main_n_passages,boost_factor=1,keep_filtered_entities=False))
pipeline_blink_alt_not_boosted_filtered.add_step(PageRankPassageRanker())
pipeline_blink_alt_not_boosted_filtered.add_step(PrintCenterEntity())
pipeline_blink_alt_not_boosted_filtered.add_step(TopNPassages(alternalive_n_passages))
pipeline_blink_alt_not_boosted_filtered.add_step(sumarizerStep)


utterances = passage_reader.data["conversation_utterance_id"]
utterances = utterances.drop_duplicates()

for utt in list(utterances[1:-1]):

    passages = passage_reader.get_utterance_passages(utt)

    print()
    print("first 3")
    out1 = pipeline_dbpedia.run(passages)
    print(out1)

    print()
    print("Top 3 of tail")
    out2 = pipeline_dbpedia_alt.run(passages)
    print(out2)

    print()
    print("head entities with boosted tail")
    out3 = pipeline_dbpedia_alt_boosted.run(passages)
    print(out3)

    print()
    print("tail entities only")
    out4 =pipeline_dbpedia_alt_not_boosted_filtered.run(passages)
    print(out4)

    print()
    print("pipeline_blink")
    out5 =pipeline_blink.run(passages)
    print(out5)

    print()
    print("blink boosted tail entities only")
    out6 =pipeline_blink_alt_not_boosted_filtered.run(passages)
    print(out6)


    print("================================")

    #if not out2 == out1:
    #    print("================================")
    #    print(out1)
    #    print("================================")
    #    print(out2)
    #    print("================================")
#