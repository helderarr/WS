from tests.BlinkReader import BlinkReader
from PassageReader import PassageReader

passage_reader = PassageReader()
passages = passage_reader.get_utterance_passages("31_1")

print(passages)

blink_reader = BlinkReader()
blink_reader.get_entities(passages)
