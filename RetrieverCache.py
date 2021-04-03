import os
import pickle

class RetrieverCache:

    def __init__(self, filename, auto_save_level=30):
        self.filename = filename
        self.dic = {}
        self.auto_save_counter = 0
        self.auto_save_level = auto_save_level

        if os.path.isfile(self.filename):
            pickle_off = open(self.filename, "rb")
            self.dic = pickle.load(pickle_off)
            pickle_off.close()

    def contains_key(self, key):
        return key in self.dic

    def get_element(self, key):
        return self.dic[key]

    def dump(self):
        pickle_off = open(self.filename, "wb")
        pickle.dump(self.dic, pickle_off)
        pickle_off.close()

    def auto_save(self, key, value):
        self.dic[key] = value

        if self.auto_save_counter >= self.auto_save_level:
            self.dump()
            self.auto_save_counter = 0
        else:
            self.auto_save_counter = self.auto_save_counter + 1
