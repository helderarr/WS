from pandas import DataFrame

from interfaces import PipelineStep


class TopNExceptPassagesBoosted(PipelineStep):

    def __init__(self, n: int, boost_factor:int = 10,keep_filtered_entities = False):
        self.n = n
        self.boost_factor = boost_factor
        self.keep_filtered_entities = keep_filtered_entities

    def run(self, data: DataFrame) -> DataFrame:
        data['int_rank'] = data['page_rank'].rank(ascending=False)

        data_top = data[data["int_rank"] <= self.n]
        data_tail = data[data["int_rank"] > self.n]

        data_top_ents, data_top_scores = self.merge_entities(data_top)
        data_tail_ents, data_tail_scores = self.merge_entities(data_tail)

        to_boost = data_tail_ents.keys() - data_top_ents.keys()

        data_tail["entities"] = data_tail["entities"].apply(lambda x: self.boost_list(x, to_boost))

        return data_tail

    def boost_list(self, entities: list, to_boost: list):
        new_entities = []
        for elem in entities:
            if elem[0] in to_boost:
                for _ in range(self.boost_factor):
                    new_entities.append(elem)
            if self.keep_filtered_entities:
                new_entities.append(elem)
        return new_entities

    def merge_entities(self, data: DataFrame):
        d = {}
        scores = {}
        for index, row in data.iterrows():
            row_entities = row['entities']
            for x in row_entities:
                d[x[0]] = x[1]
                scores[x[0]] = x[2]
        return d, scores
