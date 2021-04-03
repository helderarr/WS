import abc
from abc import ABC

from pandas import DataFrame


class PipelineStep(ABC):

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def get_output(self) -> DataFrame:
        pass

    @abc.abstractmethod
    def set_input(self, data: DataFrame):
        pass


class InitStep(PipelineStep):

    def get_output(self):
        return self.data

    def set_input(self, data):
        self.data = data

    def run(self):
        pass


class PipelineData(ABC):
    def __init__(self, data):
        self.data = data


class PassagesList():
    pass


class EntityExtractor(PipelineStep, ABC):
    pass


class PassageRanker(PipelineStep):

    def __init__(self, passages: DataFrame):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> DataFrame:
        pass


class PassageSelector(PipelineStep):

    def __init__(self, passages: DataFrame):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> DataFrame:
        pass


class PassageSummarizer(PipelineStep):

    def __init__(self, passages: DataFrame):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> DataFrame:
        pass




class PageRankPassageRanker(PassageRanker):
    pass


class TopNPassageSelector(PassageSelector):
    pass


class BartPassageSumarizer(PassageSummarizer):
    pass


class Pipeline(PipelineStep):
    steps: DataFrame = []

    def __init__(self):
        self.in_data = None
        self.out_data = None

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def set_input(self, data):
        self.in_data = data

    def run(self):
        previous_step = InitStep()
        previous_step.set_input(self.in_data)

        for step in self.steps:
            step.set_input(previous_step.get_output())
            step.run()
            previous_step = step

        self.out_data = previous_step.get_output()

    def get_output(self):
        return self.out_data


