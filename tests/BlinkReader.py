from RetrieverCache import RetrieverCache
from blink.biencoder.data_process import process_mention_data
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import blink.ner as NER
from tqdm import tqdm
import logging
import torch
import numpy as np
import json
import blink.candidate_ranking.utils as utils
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import hashlib


def get_key(data):

    sha_1 = hashlib.sha1()

    for passage in list(data["passage"]):  # Change this
        sha_1.update(passage.encode('utf-8'))

    return sha_1.hexdigest()


class BlinkReader(RetrieverCache):

    def __init__(self):
        super(BlinkReader, self).__init__(filename="data/blink.pickle")

        self.model_loaded = False
        self.evaluator = None
        self.biencoder = None
        self.biencoder_params = None
        self.candidate_encoding = None
        self.faiss_indexer = None
        self.top_k = 1
        self.logger = utils.get_logger('output')

    def get_entities(self, data):
        if self.contains_key(get_key(data)):
            return self.get_element(get_key(data))

        entities = self.load_entities(data)
        self.auto_save(get_key(data), entities)
        return entities

    def load_entities(self, data):
        if not self.model_loaded:
            self.load_model()
            self.model_loaded = True

        samples = annotate(self.ner_model, list(data["passage"]))

        dataloader = process_biencoder_dataloader(
            samples, self.biencoder.tokenizer, self.biencoder_params
        )
        labels, nns, scores = run_biencoder(
            self.biencoder, dataloader, self.candidate_encoding, self.top_k, self.faiss_indexer
        )

        print(labels, nns, scores)

        self.logger.info(nns)
        self.logger.info(scores)

        idx = 0
        for entity_list, sample in zip(nns, samples):
            e_id = entity_list[0]
            print(entity_list)
            print(sample["sent_idx"], idx, e_id)
            idx += 1

        return None

    def load_model(self):

        #self.biencoder, self.biencoder_params, self.candidate_encoding, self.faiss_indexer \
         #   = load_models(logger=self.logger)

        self.ner_model = NER.get_model()



def annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
                                 : mention["start_pos"]
                                 ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
                                  mention["end_pos"]:
                                  ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples




def process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader



def run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores



def load_candidates(entity_catalogue, entity_encoding, faiss_index=None,
                    index_path=None, logger=None):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def load_models(
        biencoder_config='/home/azureuser/test/BLINK/models/biencoder_wiki_large.json',
        biencoder_model='/home/azureuser/test/BLINK/models/biencoder_wiki_large.bin',
        entity_catalogue='/home/azureuser/test/BLINK/models/entity.jsonl',
        entity_encoding='/home/azureuser/test/BLINK/models/all_entities_large.t7',
        logger=None):
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = biencoder_model
    biencoder = load_biencoder(biencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = load_candidates(
        entity_catalogue,
        entity_encoding,
        faiss_index=None,
        index_path=None,
        logger=logger,
    )

    return biencoder, biencoder_params, candidate_encoding, faiss_indexer
