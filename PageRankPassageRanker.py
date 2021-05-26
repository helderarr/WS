from pandas import DataFrame
import pandas as pd
import numpy as np
from scipy import sparse
from operator import itemgetter

from interfaces import PipelineStep


class PageRankPassageRanker(PipelineStep):

    def run(self, data: DataFrame) -> DataFrame:
        named_entity_data = []
        heuristic = ""
        df = data[['passage', 'entities',"conversation_utterance_id", "utterance"]]
        named_entity_data = df.to_records(index=True)
        graph, order, entities = self.buildGraph(named_entity_data)
        scores = self.centrality_scores(graph)
        if heuristic == "sum":
            importancy = self.importancy_heuristic_sum(entities, scores, order)
        elif heuristic == "exp":
            importancy = self.importancy_heuristic_exp(entities, scores, order)
        elif heuristic == "nsum":
            importancy = self.importancy_heuristic_nsum(entities, scores, order)
        else:
            importancy = self.importancy_heuristic_td_idf(entities, scores, order)

        heuristics = {
            "sum": self.importancy_heuristic_sum(entities, scores, order) ,
            "exp" : self.importancy_heuristic_exp(entities, scores, order) ,
            "nsum" : self.importancy_heuristic_nsum(entities, scores, order) ,
            "td-idf" : self.importancy_heuristic_td_idf(entities, scores, order)
        }

      #  for importancy in heuristics:
       #     aux = list(heuristics.values()).index(importancy)
        #    ranked_data = self.ranked_data_builder(importancy, named_entity_data)
         #   df [ aux ] =  pd.DataFrame(ranked_data,columns=['Order', 'passage', 'page_rank', "conversation_utterance_id", "utterance"])
        importancy = self.importancy_heuristic_sum(entities, scores, order)
        ranked_data = self.ranked_data_builder(importancy, named_entity_data)
        #create DataFrame using data
        df = pd.DataFrame(ranked_data, columns=['Order', 'passage', 'page_rank',"conversation_utterance_id", "utterance"])
        return df

    def buildGraph(self, named_entity_data):
        num_respostas = len(named_entity_data)
        # hastTable
        arr = np.zeros(num_respostas)
        graph = {}
        order = []
        entities = []
        for i in range(num_respostas):
            size = len(named_entity_data[i][2])
            entities.append(named_entity_data[i][2])
            for j in range(size):
                if (named_entity_data[i][2])[j] not in graph.keys():
                    order.append((named_entity_data[i][2])[j])
                    graph[(named_entity_data[i][2])[j]] = np.zeros(num_respostas)
                (graph[(named_entity_data[i][2])[j]])[i] = 1
        graph = np.array([graph[i] for i in graph.keys()])
        graph = np.dot(graph, graph.T)
        final_graph = sparse.csr_matrix(graph)
        print('final', final_graph.shape[0])
        return final_graph, order, entities

    def centrality_scores(self, X, alpha=0.85, max_iter=100, tol=1e-10):
        """Power iteration computation of the principal eigenvector
        This method is also known as Google PageRank and the implementation
        is based on the one from the NetworkX project (BSD licensed too)
        with copyrights by:
          Aric Hagberg <hagberg@lanl.gov>
          Dan Schult <dschult@colgate.edu>
          Pieter Swart <swart@lanl.gov>
        """
        n = X.shape[0]
        X = X.copy()
        incoming_counts = np.asarray(X.sum(axis=1)).ravel()
        print("Normalizing the graph")
        for i in incoming_counts.nonzero()[0]:
            X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
        dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0),
                                     1.0 / n, 0)).ravel()
        scores = np.full(n, 1. / n, dtype=np.float32)  # initial guess
        for i in range(max_iter):
            #print("power iteration #%d" % i)
            prev_scores = scores
            scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                      + (1 - alpha) * prev_scores.sum() / n)
            # check convergence: normalized l_inf norm
            scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        #print("error: %0.6f" % err)
        if err < n * tol:
            return scores
        return scores

    def importancy_heuristic_sum(self, entities, scores, order):
        importancy = np.zeros(len(entities))
        for i in range(len(entities)):
            for j in range(len(entities[i])):
                importancy[i] += scores[order.index(entities[i][j])]
        return importancy

    def importancy_heuristic_exp(entities, scores, order):
        importancy = np.zeros(len(entities))
        for i in range(len(entities)):
            for j in range(len(entities[i])):
                importancy[i] += np.exp(scores[order.index(entities[i][j])])
        return importancy

    def importancy_heuristic_nsum(entities, scores, order):
        importancy = np.zeros(len(entities))
        for i in range(len(entities)):
            for j in range(len(entities[i])):
                importancy[i] += scores[order.index(entities[i][j])]
            importancy[i] = importancy[i] / len(entities[i])
        return importancy

    def importancy_heuristic_td_idf(entities, scores, order):
        importancy = np.zeros(len(entities))
        graph = {}
        for p in order:
            aux = 0
            for i in range(len(entities)):
                for j in range(len(entities[i])):
                    if (entities[i][j] == p):
                        aux += 1
            graph[p] = aux
        for i in range(len(entities)):
            for j in range(len(entities[i])):
                importancy[i] += scores[order.index(entities[i][j])] / graph.get(entities[i][j])

        return importancy



    def ranked_data_builder(self, importancy, named_entity_data):
        aux = []
        for i in range(len(importancy)):
            # (1 , texto , score)
            aux.append((i + 1, named_entity_data[i][1], importancy[i],named_entity_data[i][3],named_entity_data[i][4]))
        #aux.sort(key=itemgetter(2), reverse=True)
        return aux
