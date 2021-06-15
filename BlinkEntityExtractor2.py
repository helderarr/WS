from pandas import DataFrame

from BlinkReader import BlinkReader
from BlinkReaderExtended import BlinkReaderExtended
from interfaces import PipelineStep
import itertools


def group_data(column, list):
    out = []
    begin = 0
    for i in range(len(column)):
        num_name_ent = len(column[i])
        end = num_name_ent + begin
        out.append(list[begin:end])
        begin = end
    return out


class BlinkEntityExtractor2(PipelineStep):

    def __init__(self):
        super(BlinkEntityExtractor2, self).__init__()
        self.blink = BlinkReaderExtended()

    def run(self, data: DataFrame) -> DataFrame:
        column = list(data["blink_entity_in"])
        query = list(itertools.chain.from_iterable(column))
        out = self.blink.get(query)
        self.blink.dump()
        grouped_data = group_data(column, out)
        data["entities"] = grouped_data
        return data
