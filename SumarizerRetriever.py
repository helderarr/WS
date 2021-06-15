from RetrieverCache import RetrieverCache
from transformers import pipeline
import torch


class SumarizerRetriever(RetrieverCache):

    def __init__(self, min_length=50, max_length=80):
        super(SumarizerRetriever, self).__init__("data/SumarizerRetriever.piclke")
        self.min_length = min_length
        self.max_length = max_length

    def extract_element_from_source(self, key):
        if len(key) <= 0:
            return ""

        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", device=device)
        summarized = summarizer(key, min_length=min(len(key),self.min_length), max_length=min(len(key),self.max_length))
        return summarized

    def compute_key(self, key):
        return f"({self.min_length},{self.max_length}){key}"
