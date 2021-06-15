from dataclasses import dataclass
from scipy.sparse import csr_matrix
from sknetwork.ranking import PageRank
from sknetwork.hierarchy import Paris
import numpy as np
from sknetwork.visualization import svg_graph, svg_dendrogram
import math


@dataclass
class Arc:
    passage: str
    entity_id: int
    entity_name: str
    entity_text: int
    weight: float


class DirectGraph:

    def __init__(self):
        self.origin_names = []
        self.destin_idx = []
        self.destin_names = []
        self.a = []
        self.b = []
        self.v = []


    def add_arc(self, arc: Arc):
        if arc.passage not in self.origin_names:
            self.origin_names.append(arc.passage)
        if arc.entity_id not in self.destin_idx:
            self.destin_idx.append(arc.entity_id)
            self.destin_names.append(arc.entity_name)

        o_idx = self.origin_names.index(arc.passage)
        d_idx = self.destin_idx.index(arc.entity_id)

        self.a.append(o_idx)
        self.b.append(d_idx)
        self.v.append(arc.weight)

    def compute_rank(self, file_name):
        x = csr_matrix((self.v, (self.b, self.a)), shape=(len(self.destin_idx), len(self.destin_idx)), dtype=float)
        print(x)
        adjacency = x.multiply(x.transpose())
        pagerank = PageRank()
        scores = pagerank.fit_transform(adjacency)
        image = svg_graph(adjacency, names=self.destin_names, scores=scores, display_node_weight=True, node_order=np.argsort(scores))
        with open(file_name, "w") as text_file:
            print(file_name)
            print(scores)
            text_file.write(image)

        print(self.v)
        print(self.destin_names)

        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)

        image = svg_dendrogram(dendrogram, self.destin_names, n_clusters=5, rotate=True)
        with open("dento_" +file_name, "w") as text_file:
            text_file.write(image)




