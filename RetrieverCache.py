import abc
import os
import pickle
from abc import ABC


class RetrieverCache(ABC):

    def __init__(self, filename, auto_save_level=30):
        self.filename = filename
        self.dic = {}
        self.auto_save_counter = 0
        self.auto_save_level = auto_save_level

        if os.path.isfile(self.filename):
            with open(self.filename, "rb") as pickle_off:
                self.dic = pickle.load(pickle_off)

    def get(self, key):

        computed_key = self.compute_key(key)

        if not self._contains_key(computed_key):
            element = self.extract_element_from_source(key)
            self._auto_save(computed_key, element)
        else:
            element = self._get_element(computed_key)
        return element

    def _contains_key(self, key):
        return key in self.dic

    def _get_element(self, key):
        return self.dic[key]

    def dump(self):
        if self.auto_save_counter > 0:
            with open(self.filename, "wb") as pickle_off:
                pickle.dump(self.dic, pickle_off)
            self.auto_save_counter = 0

    def _auto_save(self, key, value):
        self.dic[key] = value

        if self.auto_save_counter >= self.auto_save_level:
            self.dump()
        else:
            self.auto_save_counter = self.auto_save_counter + 1

    def compute_key(self, key):
        return key

    @abc.abstractmethod
    def extract_element_from_source(self, key):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dump()
