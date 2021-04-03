from ElasticSearchSimpleAPI import ESSimpleAPI
from RetrieverCache import RetrieverCache


class PassageRetriever(RetrieverCache):

    def __init__(self):
        super(PassageRetriever, self).__init__(filename="data/db.pickle")
        self.es = ESSimpleAPI()

    def get_passage(self, doc_id):
        if self.contains_key(doc_id):
            return self.get_element(doc_id)

        try:
            passage = self.es.get_doc_body(doc_id)
            self.auto_save(doc_id, passage)
            return passage
        except Exception as e:
            print(e)
            return None