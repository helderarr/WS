from elasticsearch import Elasticsearch
from pandas import json_normalize


class ESSimpleAPI:

    def __init__(self):
        self.client = Elasticsearch('https://elasticsearch:muAwikri@api.novasearch.org/elasticsearch/v7.4/')
        self.client.indices.open('msmarco')
        self.index="msmarco"
        
    def search_body(self, query=None, numDocs=10):
        result = self.client.search(index=self.index, body={"query": {"match": {"body": {'query': query}}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    ####################################################################################
    #### DELETE RM3
    def search_body_RM3(self, query=None, numDocs=10, expDocs=10, expTerms=10):
        result = self.client.search(index=self.index, body={"query": {"match": {"body": {'query': query}}}}, size=expDocs)
        df = json_normalize(result["hits"]["hits"])
        
        result = self.client.search(index=self.index, body={"query": {"match": {"body": {'query': expanded_query}}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        
        return df
    #### DELETE RM3
    ####################################################################################

    
    def search_entities(self, query=None, numDocs=10):
        result = self.client.search(index=self.index, body={"query": {"entities": {"body": {'query': query}}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def search_QSL(self, query_qsl=None, numDocs=10):
        result = self.client.search(index=self.index, body=query_qsl, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def doc_term_vectors(self, doc_id):
        term_statistics_json = self.client.termvectors(index='msmarco', id=doc_id, fields='body', field_statistics="true", term_statistics ='true')
        doc_freq = term_statistics_json["term_vectors"]["body"]["field_statistics"]["doc_count"]
        sum_doc_freq = term_statistics_json["term_vectors"]["body"]["field_statistics"]["sum_doc_freq"]
        sum_ttf = term_statistics_json["term_vectors"]["body"]["field_statistics"]["sum_ttf"]
        term_statistics={}
        for term in term_statistics_json["term_vectors"]["body"]["terms"]:
            term_statistics[term] = [term_statistics_json["term_vectors"]["body"]["terms"][term]["term_freq"], term_statistics_json["term_vectors"]["body"]["terms"][term]["doc_freq"], term_statistics_json["term_vectors"]["body"]["terms"][term]["ttf"]]
        return doc_freq, sum_doc_freq, sum_ttf, term_statistics

    def multi_doc_term_vectors(self, doc_ids):
        term_statistics_json = self.client.mtermvectors(index='msmarco', ids=doc_ids, fields='body', field_statistics="true", term_statistics ='true')
        docs_term_vectors={}

        doc_freq = term_statistics_json["docs"][0]["term_vectors"]["body"]["field_statistics"]["doc_count"]
        sum_doc_freq = term_statistics_json["docs"][0]["term_vectors"]["body"]["field_statistics"]["sum_doc_freq"]
        sum_ttf = term_statistics_json["docs"][0]["term_vectors"]["body"]["field_statistics"]["sum_ttf"]

        for doc in term_statistics_json["docs"]:
            term_statistics={}
            doc_id = doc["_id"]
            for term in doc["term_vectors"]["body"]["terms"]:
                term_statistics[term] = [doc["term_vectors"]["body"]["terms"][term]["term_freq"], doc["term_vectors"]["body"]["terms"][term]["doc_freq"], doc["term_vectors"]["body"]["terms"][term]["ttf"]]
            docs_term_vectors[doc_id] = term_statistics

        return doc_freq, sum_doc_freq, sum_ttf, docs_term_vectors
    
    def query_terms(self, query, analyzer):
        tokens = self.client.indices.analyze(index = 'msmarco', body = {"analyzer": analyzer, "text": query})
        norm_query = ""
        for term in tokens['tokens']: 
            norm_query = norm_query + " " + term['token']

        return norm_query
    
    def get_doc_body(self, doc_id):
        aa = self.client.get(index="msmarco", id=doc_id)["_source"]["body"]
        return aa

