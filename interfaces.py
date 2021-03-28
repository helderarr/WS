import abc
from dataclasses import dataclass


class PipelineStep(abc):
    def run(self):
        pass

class PipelineData(abc):
    def __init__(self,data):
        self.data = data

class PassagesList():
    pass

class EntityExtractor(PipelineStep):

    def __init__(self, passages: list[str]):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> list[(str, list[int])]:
        pass


class PassageRanker(PipelineStep):

    def __init__(self, passages: list[str]):
        self.passages = passages

    def run(self):
        pass

    def get_output(self) -> list[(str, list[int])]:
        pass
