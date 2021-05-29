from RetrieverCache import RetrieverCache
from transformers import pipeline
import torch


class SumarizerRetriever(RetrieverCache):

    def __init__(self, min_length=100, max_length=180):
        super(SumarizerRetriever, self).__init__("data/SumarizerRetriever.piclke")
        self.min_length = min_length
        self.max_length = max_length

    def extract_element_from_source(self, key):
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", device=device)
        summarized = summarizer(key, min_length=100, max_length=180)
        return summarized

    def compute_key(self, key):
        return f"({self.min_length},{self.max_length}){key}"
