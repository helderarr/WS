from pandas import DataFrame

from interfaces import PipelineStep


class PrintCenterEntity(PipelineStep):
    def run(self, data: DataFrame) -> DataFrame:

        ents, scores = self.merge_entities(data)

        if len(ents) == 0:
            print("No entities found.")
            return data

        max_score = 0
        entity_with_max_score = None
        for ent in ents:
            if scores[ent] > max_score:
                entity_with_max_score = ent
                max_score = scores[ent]

        print("Center Entity:",ents[entity_with_max_score])

        return data


    def merge_entities(self,data:DataFrame):
        d ={}
        scores ={}
        for index, row in data.iterrows():
            row_entities = row['entities']
            for x in row_entities:
                d[x[0]] = x[1]
                scores[x[0]] = x[2]

        return d,scores

