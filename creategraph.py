import pickle

filename = "data/BlinkReaderExtended.pickle"

with open(filename, "rb") as pickle_off:
    data = pickle.load(pickle_off)
