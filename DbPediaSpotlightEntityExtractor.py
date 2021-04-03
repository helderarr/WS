from abc import ABC
import requests
from pandas import DataFrame

from interfaces import EntityExtractor


class DbPediaSpotlightEntityExtractor(EntityExtractor):

    def set_input(self, data:DataFrame):
        self.data = data

    def run(self):
        self.data["entities"] = self.data.apply(lambda row: self.extract_single_sentence(row["passage"]))

    def extract_single_sentence(self, text:str):
        url = "https://api.dbpedia-spotlight.org/en/annotate"
        headers = {"Accept": "application/json"}
        params = {"text": text}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_output(self):
        return self.data


