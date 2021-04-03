

from ElasticSearchSimpleAPI import ESSimpleAPI
from RetrieverCache import RetrieverCache


class PassageRetriever(RetrieverCache):

    def extract_element_from_source(self, key):
        try:
            passage = self.es.get_doc_body(key)
            return passage
        except Exception as e:
            print(e)
            return None

    def __init__(self):
        super(PassageRetriever, self).__init__(filename="data/db.pickle")
        self.es = ESSimpleAPI()


