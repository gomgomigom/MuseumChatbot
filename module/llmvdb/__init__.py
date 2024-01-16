from typing import Optional, Literal
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB
from .customdataset import CustomDataset, EvalCustomDataset
from tqdm.auto import tqdm
from .doc import ToyDoc
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from .bm25 import CustomBM25


class Llmvdb:
    def __init__(
        self,
        embedding=None,
        llm=None,
        verbose: bool = False,
        file_path=None,
        workspace: Optional[str] = None,
        threshold: float = 0.7,
        top_k: int = 3,
        bm25: Optional[Literal["bert", "space"]] = None,
    ):
        self.embedding = embedding
        self.llm = llm
        self.verbose = verbose
        self.workspace = workspace
        self.file_path = file_path
        self.threshold = threshold
        self.top_k = top_k
        self.bm25 = None
        if bm25:
            self.bm25 = CustomBM25(tokenizer=bm25)
            self.bm25_tok = bm25
        self.db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

    def custom_collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        questions = [item["question"] for item in batch]
        ctx_ids = [item["ctx_id"] for item in batch]
        tit_ids = [item["tit_id"] for item in batch]
        batched_data = []
        for i in range(len(batch)):
            batched_data.append(
                {
                    "text": texts[i],
                    "question": questions[i],
                    "ctx_id": ctx_ids[i],
                    "tit_id": tit_ids[i],
                }
            )
        return batched_data

    def initialize_db(self):
        dataset = CustomDataset(self.file_path)
        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, collate_fn=self.custom_collate_fn
        )
        doc_list = []

        for batch in tqdm(dataloader, desc="Processing dataset embedding"):
            texts = [data["text"] for data in batch if data["text"].strip() != ""]
            context_embeddings = self.embedding.get_embedding(texts)

            for j, data in enumerate(batch):
                if data["text"].strip() != "":
                    doc_list.append(
                        ToyDoc(
                            text=data["text"],
                            context_embedding=context_embeddings[j],
                            question=data["question"],
                            tit_id=data["tit_id"],
                            ctx_id=data["ctx_id"],
                        )
                    )

        self.db.index(inputs=DocList[ToyDoc](doc_list))
        self.db.persist()

    def retrieve_document(self, prompt):
        input_document = ""
        if self.bm25:
            if self.verbose:
                print(f"=====bm25_{self.bm25_tok}를 사용하여 문서를 {self.top_k}개 검색합니다=====")
            result = self.bm25.get_top_n(prompt, self.top_k)
            for i, doc in enumerate(result):
                input_document += "#문서" + str(i) + "\n" + doc + "\n"
        else:
            query = ToyDoc(
                text=prompt, context_embedding=self.embedding.get_embedding(prompt)
            )
            search_parameters = {"search_field": "context_embedding"}
            results = self.db.search(
                inputs=DocList[ToyDoc]([query]),
                parameters=search_parameters,
                limit=self.top_k,
            )

            over_threshold_indices = [
                idx
                for idx, value in enumerate(results[0].scores)
                if value > self.threshold
            ]

            if self.verbose:
                # print(results[0].matches[0])
                # print(results[0].matches.ctx_id)
                print(results[0].text, results[0].scores)
                print(f"threshold를 넘는 index : {over_threshold_indices}")

                # 만약 threshold 0.8을 넘는게 있고 그 개수가 k개보다 적다면 전부 retrieve
            if 1 <= len(over_threshold_indices) < self.top_k:
                for index in over_threshold_indices:  # top-k (k=3)
                    input_document += (
                        "#문서"
                        + str(index)
                        + "\n"
                        + results[0].matches[index].text
                        + "\n"
                    )

            # 만약 threshold 0.8을 넘는게 있고 그 개수가 k개보다 많다면 top-k만 retrieve
            elif len(over_threshold_indices) >= self.top_k:
                for index in range(self.top_k):  # top-k (k=3)
                    input_document += (
                        "#문서"
                        + str(index)
                        + "\n"
                        + results[0].matches[index].text
                        + "\n"
                    )

            # 만약 threshold 0.8을 넘는게 없다면 top-1만
            elif len(over_threshold_indices) == 0:
                input_document += "#문서\n" + results[0].matches[0].text + "\n"

        if self.verbose:
            print("================아래 문서를 참고합니다================")
            print(input_document)
            print("======================================================")

        return input_document

    def generate_response(self, prompt):
        input_document = self.retrieve_document(prompt)
        completion = self.llm.call(prompt, input_document)
        return completion

    def change_embedding(self, new_embedding):
        self.embedding = new_embedding

    def evaluate_model(self, target_model):
        dataset = EvalCustomDataset("data/test.jsonl")
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=False, collate_fn=self.custom_collate_fn
        )
        question_list = []
        for batch in tqdm(dataloader, desc=f"embedding..{target_model}"):
            question = [data["question"] for data in batch]
            question_embedding = self.embedding.get_embedding(question)
            for idx, data in enumerate(batch):
                question_list.append(
                    ToyDoc(
                        question=data["question"],
                        question_embedding=question_embedding[idx],
                        ctx_id=data["ctx_id"],
                        tit_id=data["tit_id"],
                    )
                )
        question_len = len(question_list)
        correct_counts = {
            1: 0,
            2: 0,
            3: 0,
            5: 0,
            7: 0,
            10: 0,
            20: 0,
            50: 0,
            "model": target_model,
            "criteria": "ctx_id",
        }
        correct_counts_tit_id = correct_counts.copy()
        correct_counts_tit_id["criteria"] = "tit_id"

        top_k_keys = sorted([k for k in correct_counts.keys() if isinstance(k, int)])

        for q in tqdm(question_list, desc=f"search..{target_model}"):
            search_query = ToyDoc(
                question=q.question,
                context_embedding=q.question_embedding,
                ctx_id=q.ctx_id,
                tit_id=q.tit_id,
            )
            for top_k in top_k_keys:
                search_parameters = {"search_field": "context_embedding"}
                search_results = self.db.search(
                    inputs=DocList[ToyDoc]([search_query]),
                    parameters=search_parameters,
                    limit=top_k,
                )

                ctx_id_matched = False
                for match in search_results[0].matches:
                    if match.ctx_id == search_query.ctx_id:
                        correct_counts[top_k] += 1
                        correct_counts_tit_id[top_k] += 1
                        ctx_id_matched = True
                        break

                if not ctx_id_matched:
                    for match in search_results[0].matches:
                        if match.tit_id == search_query.tit_id:
                            correct_counts_tit_id[top_k] += 1
                            break

        for key in top_k_keys:
            correct_counts[key] = correct_counts[key] / question_len
            correct_counts_tit_id[key] = correct_counts_tit_id[key] / question_len

        return correct_counts, correct_counts_tit_id
