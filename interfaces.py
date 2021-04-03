import abc
from abc import ABC

from pandas import DataFrame


class PipelineStep(ABC):

    def __init__(self):
        self.data = None

    @abc.abstractmethod
    def run(self, data: DataFrame) -> DataFrame:
        pass


class Pipeline(PipelineStep):

    def __init__(self):
        self.steps: DataFrame = []
        pass

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def run(self, data: DataFrame) -> DataFrame:
        previous_data = data
        for step in self.steps:
            next_data = step.run(previous_data)
            previous_data = next_data

        return next_data
