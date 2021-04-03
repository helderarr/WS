from abc import ABC
import requests
from pandas import DataFrame

from interfaces import PipelineStep


class DbPediaSpotlightEntityExtractor(PipelineStep):

    def run(self, data: DataFrame) -> DataFrame:
        data["entities"] = data.apply(lambda row: extract_single_sentence(row["passage"]), axis=1)
        return data

def extract_single_sentence(text: str):
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
