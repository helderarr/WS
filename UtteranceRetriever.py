import pandas as pd


class UtteranceRetirever:

    def __init__(self):
        self.file_name = "data/evaluation_topics_annotated_resolved_v1.0.tsv"
        file = pd.read_csv(self.file_name, names=["conversation_utterance_id", "utterance"], delimiter="\t")
        self.data = file.set_index("conversation_utterance_id").T.to_dict("list")

    def get_utterance(self, conversation_utterance_id):
        return self.data[conversation_utterance_id][0]



