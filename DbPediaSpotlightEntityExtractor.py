from abc import ABC
import requests
from pandas import DataFrame

from DbPediaEntityRetriever import DbPediaEntityRetriever
from interfaces import PipelineStep


class DbPediaSpotlightEntityExtractor(PipelineStep):
    def __init__(self):
        self.extractor = DbPediaEntityRetriever()

    def run(self, data: DataFrame) -> DataFrame:
        data["entities"] = data.apply(lambda row: self.extract_single_sentence(row["passage"]), axis=1)
        return data

    def extract_single_sentence(self,text: str):
        entity = self.extractor.get(text)
        if entity is None:
            return []
        return entity
