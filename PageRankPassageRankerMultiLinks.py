from pandas import DataFrame
import pandas as pd
import numpy as np
from scipy import sparse
from operator import itemgetter
from os import path
from interfaces import PipelineStep
import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from openpyxl import load_workbook


class PageRankPassageRankerMultiLinks(PipelineStep):

    def __init__(self, file_name):
        super(PageRankPassageRankerMultiLinks, self).__init__()
        self.file_name = file_name

    def run(self, data: DataFrame) -> DataFrame:
        named_entity_data = []
        df = data[['passage', 'entities', "conversation_utterance_id", "utterance"]]
        named_entity_data = df.to_records(index=True)
        graph, order, entities, names, first_graph = self.buildGraph(named_entity_data)
        scores = self.centrality_scores(graph)

        X = np.array(scores).reshape(-1, 1)
        clustering = KMeans(n_clusters=int(len(scores) / 5)).fit(X)
        print(clustering.labels_)

        names_list = [names[x] for x in order]

        first_graph[first_graph == 0] = 1/1000
        inv_first_graph = 1 / first_graph
        clustering_line = AffinityPropagation(damping=0.75).fit(inv_first_graph)

        centers = [x in clustering_line.cluster_centers_indices_ for x in range(len(scores))]

        test_df = pd.DataFrame({
            'names': names_list,
            'scores': scores,
            'cluster': clustering.labels_,
            'centers':centers,
            'line_cluster': clustering_line.labels_
        })
        test_df = test_df.sort_values(by=['scores'], ascending=False)

        if path.exists(self.file_name):
            book = load_workbook(self.file_name)
            writer = pd.ExcelWriter(self.file_name, engine='openpyxl')
            writer.book = book
        else:
            writer = pd.ExcelWriter(self.file_name, engine='openpyxl')


        utt = df["conversation_utterance_id"].array[0]
        df.to_excel(writer, sheet_name=f"{utt}_P")
        test_df.to_excel(writer, sheet_name=f"{utt}_G")

        writer.save()
        writer.close()

        importancy = self.importancy_heuristic_sum(entities, scores, order)
        ranked_data = self.ranked_data_builder(importancy, named_entity_data)
        # create DataFrame using data
        df = pd.DataFrame(ranked_data,
                          columns=['Order', 'passage', 'page_rank', "conversation_utterance_id", "utterance"])
        return df

    def buildGraph(self, named_entity_data):
        num_respostas = len(named_entity_data)
        # hastTable
        arr = np.zeros(num_respostas)
        graph = {}
        order = []
        entities = []
        names_dic = {}
        for i in range(num_respostas):
            data, scores, names, desc = self.get_entities(named_entity_data[i][2])
            size = len(data)
            entities.append(data)
            for j in range(size):
                if data[j] not in graph.keys():
                    order.append(data[j])
                    graph[data[j]] = np.zeros(num_respostas)
                (graph[data[j]])[i] = scores[j]

            for k, v in names.items():
                if k not in names_dic:
                    names_dic[k] = v

        graph = np.array([graph[i] for i in graph.keys()])
        graph_prod = np.dot(graph, graph.T)
        final_graph = sparse.csr_matrix(graph_prod)
        print('final', final_graph.shape[0])
        return final_graph, order, entities, names_dic, graph

    def get_entities(self, data):
        names = {}
        desc = {}

        entity_groups = list(itertools.chain.from_iterable([list(x[0]) for x in data]))
        score_groups = list(itertools.chain.from_iterable([list(x[1]) for x in data]))
        name_groups = list(itertools.chain.from_iterable([list(x[2]) for x in data]))
        desc_groups = list(itertools.chain.from_iterable([list(x[3]) for x in data]))

        for i in range(len(entity_groups)):
            if entity_groups[i] not in names.keys():
                names[entity_groups[i]] = name_groups[i]
                desc[entity_groups[i]] = desc_groups[i]

        return entity_groups, score_groups, names, desc

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
            # print("power iteration #%d" % i)
            prev_scores = scores
            scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                      + (1 - alpha) * prev_scores.sum() / n)
            # check convergence: normalized l_inf norm
            scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        # print("error: %0.6f" % err)
        if err < n * tol:
            return scores
        return scores

    def importancy_heuristic_sum(self, entities, scores, order):
        importancy = np.zeros(len(entities))
        for i in range(len(entities)):
            for j in range(len(entities[i])):
                importancy[i] += scores[order.index(entities[i][j])]
        return importancy

    def ranked_data_builder(self, importancy, named_entity_data):
        aux = []
        for i in range(len(importancy)):
            # (1 , texto , score)
            aux.append(
                (i + 1, named_entity_data[i][1], importancy[i], named_entity_data[i][3], named_entity_data[i][4]))
        # aux.sort(key=itemgetter(2), reverse=True)
        return aux
