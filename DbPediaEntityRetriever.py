import requests

from RetrieverCache import RetrieverCache


class DbPediaEntityRetriever(RetrieverCache):

    def __init__(self):
        super(DbPediaEntityRetriever, self).__init__("data/dbpediaentity.pickle")

    def extract_element_from_source(self, text):
        try:
            url = "https://api.dbpedia-spotlight.org/en/annotate"
            headers = {"Accept": "application/json"}
            params = {"text": text}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            ids = [(x["@support"] ,x["@surfaceForm"] ) for x in response.json()['Resources']]
            return ids
        except:
            print("Error:", text)
            return None

