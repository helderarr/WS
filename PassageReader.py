from os import path
from PassageRetriever import PassageRetriever
import pandas as pd


class PassageReader:

    def __init__(self):
        self.file_name = "data/dataset.csv"
        self.data = self.load_data()

    def load_data(self):
        if not path.exists(self.file_name):
            self.extract_data()

        return pd.read_csv(self.file_name,
                           names=["conversation_utterance_id", "conversation_id", "utterance_id",
                                  "rank", "score", "passage"])

    def get_utterance_passages(self, conversation_utterance_id: str):
        return self.data[self.data["conversation_utterance_id"] == conversation_utterance_id]

    def extract_data(self):
        with PassageRetriever() as retriever:
            df = pd.read_csv('data/anserini_test_lmd_1000_car_marco_wapo_2019_bert_1000_CORIG.run',
                             names=["conversation_utterance_id", "NA", "passage_id", "gobal_rank", "score", "dataset"],
                             header=None, delimiter=" ")

            marco_df = df[df["passage_id"].str.startswith('MARCO_', na=False)]

            marco_df['rank'] = marco_df.groupby('conversation_utterance_id')['gobal_rank'].rank(method='first')

            top_10_df = marco_df[marco_df["rank"] <= 10]

            top_10_df["passage"] = top_10_df.apply(lambda row: retriever.get(row["passage_id"]), axis=1)

            top_10_df["conversation_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[0],
                                                           axis=1)

            top_10_df["utterance_id"] = top_10_df.apply(lambda row: row["conversation_utterance_id"].split("_")[1],
                                                        axis=1)

            top_10_df.to_csv(self.file_name, header=True, index=False,
                             columns=["conversation_utterance_id", "conversation_id", "utterance_id", "rank", "score",
                                      "passage"])
