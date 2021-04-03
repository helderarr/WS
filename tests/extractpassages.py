from PassageRetriever import PassageRetriever
import pandas as pd

retriever = PassageRetriever()

df = pd.read_csv('../data/anserini_test_lmd_1000_car_marco_wapo_2019_bert_1000_CORIG.run',
                 names=["conversation_utterance_id", "NA", "passage_id", "gobal_rank", "score", "dataset"],
                 header=None, delimiter=" ")

marco_df = df[df["passage_id"].str.startswith('MARCO_', na=False)]

marco_df['rank'] = marco_df.groupby('conversation_utterance_id')['gobal_rank'].rank(method='first')

top_10_df = marco_df[marco_df["rank"] <= 10]

top_10_df["passage"] = top_10_df.apply(lambda row: retriever.get_passage(row["passage_id"]), axis=1)

top_10_df["conversation_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[0], axis=1)

top_10_df["utterance_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[1], axis=1)

retriever.dump()

top_10_df.to_csv("data/dataset.csv", header=True, index=False,
                 columns=["conversation_utterance_id", "conversation_id", "utterance_id", "rank", "score", "passage"])

print(top_10_df)
