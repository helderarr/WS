from pandas import DataFrame

from NerRetreiver import NerRetriever
from interfaces import PipelineStep


class DbPediaSpotlightAnnotator(PipelineStep):

    def __init__(self):
        super(DbPediaSpotlightAnnotator, self).__init__()

    def run(self, data: DataFrame) -> DataFrame:
        with NerRetriever() as ner:
            data["blink_entity_in"] = data.apply(lambda r: get_blink_obj(ner, r["passage"]), axis=1)
            return data


def format_obj(ner, text: str, id):
    try:

        name = ner["@name"]
        offset = int(ner["@offset"])

        obj = {
            "id": id,
            "label": "unknown",
            "label_id": -1,
            "context_left": text[0:offset].strip().lower(),
            "mention": name.lower(),
            "context_right": text[offset + len(name):-1].strip().lower()
        }

    except:
        return None

    return obj


def get_blink_obj(ner, passage):
    ner_data = ner.get(passage)

    if isinstance(ner_data, list):
        data = [format_obj(x, passage, n) for x, n in zip(ner_data, range(0, len(ner_data)))]
    else:
        data = [format_obj(ner_data, passage, 0)]

    return data

