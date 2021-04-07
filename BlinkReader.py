import argparse
import hashlib
import numpy as np
from RetrieverCache import RetrieverCache
from blink import main_dense


class BlinkReader(RetrieverCache):

    def __init__(self):
        super(BlinkReader, self).__init__(filename="data/blink.pickle")

        self.models = None
        self.args = None
        self.models_path = "/home/azureuser/test/BLINK/models/"  # the path where you stored the BLINK models

        self.config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 1,
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

        return list(data)

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

        return np.concatenate(nns, axis=None)

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
