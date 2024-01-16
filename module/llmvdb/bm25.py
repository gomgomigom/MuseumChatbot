import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rank_bm25 import BM25Okapi
import numpy as np
from llmvdb.customdataset import CustomDataset, EvalCustomDataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Literal

bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


def tokenizer_space(doc):
    if isinstance(doc, list):
        return doc
    sw = set()
    with open("data/stopwords-ko.txt", encoding="utf-8") as f:
        for word in f:
            sw.add(word.strip())
    return [word for word in doc.split() if word not in sw and len(word) > 1]


def tokenizer_bert(doc):
    if isinstance(doc, list):
        return doc
    tokens = bert_tokenizer.tokenize(doc)
    return tokens


class CustomBM25(BM25Okapi):
    def __init__(
        self,
        corpus=CustomDataset("data/train.jsonl").get_all_data(),
        tokenizer: Literal["bert", "space"] = "space",
    ):
        if tokenizer == "bert":
            self.tokenizer = tokenizer_bert
        elif tokenizer == "space":
            self.tokenizer = tokenizer_space
        else:
            raise ValueError("tokenizer is 'space' or 'bert'")
        self.corpus = corpus  # corpus 속성 추가
        super().__init__(
            [self.tokenizer(doc["text"]) for doc in corpus], self.tokenizer
        )

    def get_top_n(self, query, n=10):
        tokenized_query = self.tokenizer(query)
        scores = self.get_scores(tokenized_query)
        top_docs_indices = np.argsort(scores)[::-1][:n]
        return [self.corpus[i]["text"] for i in top_docs_indices]

    def calculate_accuracy(
        self, dataset, k=10, id_key: Literal["ctx_id", "tit_id"] = "ctx_id"
    ):
        if id_key not in ["ctx_id", "tit_id"]:
            raise ValueError("id_key is 'ctx_id' or 'tit_id'")
        correct_predictions = 0
        total_questions = len(dataset)

        for item in dataset:
            question = item["question"]
            id_value = item[id_key]
            tokenized_question = self.tokenizer(question)
            scores = self.get_scores(tokenized_question)
            top_n_indices = np.argsort(scores)[::-1][:k]
            if any(self.corpus[index][id_key] == id_value for index in top_n_indices):
                correct_predictions += 1

        accuracy = correct_predictions / total_questions
        return accuracy
