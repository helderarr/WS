from pandas import DataFrame

from SumarizerRetriever import SumarizerRetriever
from interfaces import PipelineStep


class SumarizerStep(PipelineStep):

    def __init__(self, min_length=100, max_length=180):
        self.sumarizer = SumarizerRetriever(min_length, max_length)

    def run(self, data: DataFrame) -> DataFrame:
        text = ""
        for line in list(data["passage"]):
            text += line + "\n"

        summary = self.sumarizer.get(text)

        self.sumarizer.dump()

        if data.shape[0] <= 0:
            return "","",""

        return data["conversation_utterance_id"].iloc[0], data["utterance"].iloc[0], summary[0]["summary_text"]


