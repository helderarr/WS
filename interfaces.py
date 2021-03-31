import abc
from dataclasses import dataclass

from numpy import iterable


class PipelineStep(abc):

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def get_output(self) -> iterable:
        pass

    @abc.abstractmethod
    def set_input(self, data: iterable):
        pass


class InitStep(PipelineStep):

    def get_output(self):
        return self.data

    def set_input(self, data):
        self.data = data


class PipelineData(abc):
    def __init__(self, data):
        self.data = data


class PassagesList():
    pass


class EntityExtractor(PipelineStep):

    def run(self):
        pass


class PassageRanker(PipelineStep):

    def __init__(self, passages: list[str]):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> list[(str, list[int])]:
        pass


class PassageSelector(PipelineStep):

    def __init__(self, passages: list[str]):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> list[(str, list[int])]:
        pass


class PassageSummarizer(PipelineStep):

    def __init__(self, passages: list[str]):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> list[(str, list[int])]:
        pass


class BlinkEntityExtractor(EntityExtractor):
    pass


class PageRankPassageRanker(PassageRanker):
    pass


class TopNPassageSelector(PassageSelector):
    pass


class BartPassageSumarizer(PassageSummarizer):
    pass


class Pipeline(PipelineStep):
    steps: list[PipelineStep] = []

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


pipeline = Pipeline()
pipeline.add_step(BlinkEntityExtractor())
pipeline.add_step(PageRankPassageRanker())
pipeline.add_step(TopNPassageSelector())
pipeline.add_step(BartPassageSumarizer())

pipeline.in_data()
