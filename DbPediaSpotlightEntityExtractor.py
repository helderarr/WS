from abc import ABC
import requests
from pandas import DataFrame

from interfaces import EntityExtractor


class DbPediaSpotlightEntityExtractor(EntityExtractor):

    def __init__(self):
        self.data = None

    def set_input(self, data: DataFrame):
        self.data = data

    def run(self):
        self.data["entities"] = self.data.apply(lambda row: self.extract_single_sentence(row["passage"]), axis=1)

    def extract_single_sentence(self, text: str):
        try:
            url = "https://api.dbpedia-spotlight.org/en/annotate"
            headers = {"Accept": "application/json"}
            params = {"text": text}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            ids = [x["@support"] for x in response.json()['Resources']]
            return ids
        except:
            print("Error:", text)
            return None

    def get_output(self):
        return self.data
