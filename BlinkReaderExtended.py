import argparse
import hashlib
import numpy as np
from RetrieverCache import RetrieverCache
from blink import main_dense


class BlinkReaderExtended(RetrieverCache):

    def __init__(self):
        super(BlinkReaderExtended, self).__init__(filename="data/BlinkReaderExtended.pickle")

        self.models = None
        self.args = None
        self.models_path = "/home/azureuser/blink/BLINK/models/"  # the path where you stored the BLINK models

        self.config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 3,
            "biencoder_model": self.models_path + "biencoder_wiki_large.bin",
            "biencoder_config": self.models_path + "biencoder_wiki_large.json",
            "entity_catalogue": self.models_path + "entity.jsonl",
            "entity_encoding": self.models_path + "all_entities_large.t7",
            "crossencoder_model": self.models_path + "crossencoder_wiki_large.bin",
            "crossencoder_config": self.models_path + "crossencoder_wiki_large.json",
            "fast": False,  # set this to be true if speed is a concern
            "output_path": "logs/"  # logging directory
        }

    def extract_element_from_source(self, key):
        # lazy load
        if self.models is None:
            self.args = argparse.Namespace(**self.config)
            self.models = main_dense.load_models(self.args, logger=None)

        data = self.my_run(self.args, None, *self.models, test_data=key)

        return data

    def my_run(self, args, logger, biencoder, biencoder_params, crossencoder, crossencoder_params,
               candidate_encoding, title2id, id2title, id2text, wikipedia_id2local_id,
               faiss_indexer=None, test_data=None):

        samples = test_data

        dataloader = main_dense._process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        top_k = args.top_k
        labels, nns, scores = main_dense._run_biencoder(
            biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
        )

        data = []
        for i in range(len(nns)):
            ent = nns[i]
            sco = scores[i]
            titles = [id2title[idx] for idx in ent]
            textes = [id2text[idx] for idx in ent]

            data.append((ent,sco,titles,textes))

        return data

    def compute_key(self, key):
        try:

            sha_1 = hashlib.sha1()

            for passage in key:  # Change this
                sha_1.update(passage["context_left"].encode('utf-8'))
                sha_1.update(passage["context_right"].encode('utf-8'))
                sha_1.update(passage["mention"].encode('utf-8'))

            return sha_1.hexdigest()

        except TypeError as error:
            print("error creating key")
            raise error
