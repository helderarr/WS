from pandas import DataFrame

from interfaces import PipelineStep


class TopNPassages(PipelineStep):

    def __init__(self, n: int):
        self.n = n

    def run(self, data: DataFrame) -> DataFrame:
        data['int_rank'] = data['page_rank'].rank(ascending=False)

        filtered = data[data["int_rank"] <= self.n]

        return filtered
