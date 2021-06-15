from pandas import DataFrame

from interfaces import PipelineStep


class TopNPassagesWithAlternative(PipelineStep):

    def __init__(self, n: int,interested_in_alternative=False):
        super().__init__()
        self.n = n
        self.interested_in_alternative = interested_in_alternative

    def run(self, data: DataFrame) -> DataFrame:
        data['int_rank'] = data['page_rank'].rank(ascending=False)

        filtered = data[data["int_rank"] <= self.n]

        if not self.interested_in_alternative:
            return filtered

        ents_filtered , scores_filtered= self.merge_entities(filtered)
        ents_total, scores_total = self.merge_entities(data)

        ents_not_considered = ents_total.keys() - ents_filtered.keys()

        max_score = 0
        entity_with_max_score = None
        for ent in ents_not_considered:
            if scores_total[ent[0]] > max_score:
                entity_with_max_score = ent
                max_score = scores_total[ent[0]]

        print("Also interested in",entity_with_max_score[1],"?")


        return filtered

    def merge_entities(self,data:DataFrame):
        d ={}
        scores ={}
        for index, row in data.iterrows():
            row_entities = row['entities']
            for x in row_entities:
                d[x[0]] = x[1]
                scores[x[0]] = x[2]

        return d,scores
