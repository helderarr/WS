import requests

from RetrieverCache import RetrieverCache


class NerRetriever(RetrieverCache):

    def __init__(self):
        super(NerRetriever, self).__init__(filename="data/ner.pickle")

    def extract_element_from_source(self, key):
        url = "https://api.dbpedia-spotlight.org/en/spot"
        headers = {"Accept": "application/json"}
        params = {"text": key}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        try:
            data = response.json()['annotation']['surfaceForm']
            return data
        except:
            return []
