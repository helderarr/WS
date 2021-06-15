from pandas import DataFrame
from graph import Arc, DirectGraph
from datetime import datetime
from interfaces import PipelineStep


class GraphPrint(PipelineStep):

    def run(self, data: DataFrame) -> DataFrame:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")


        data.to_excel(dt_string +"_output.xlsx")

        graph = DirectGraph()

        for index, row in data.iterrows():
            passage = row["passage"]
            for ent in row["entities"]:
                for i in range(3):
                    arc = Arc(passage, ent[0][i], ent[2][i], ent[3][i], ent[1][i])

                    graph.add_arc(arc)


        graph.compute_rank(dt_string +"_test.svg")

        return data

    def __init__(self):
        pass
