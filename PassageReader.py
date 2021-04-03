import pandas as pd


class PassageReader:

    def __init__(self):
        self.data = self.load_data()

    @staticmethod
    def load_data():
        return pd.read_csv("data/dataset.csv",
                           names=["conversation_utterance_id", "conversation_id", "utterance_id",
                                    "rank", "score", "passage"])

    def get_utterance_passages(self, conversation_utterance_id: str):
        return self.data[self.data["conversation_utterance_id"] == conversation_utterance_id]
