from typing import Optional
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB
from .customdataset import CustomDataset
from tqdm.auto import tqdm
from .doc import ToyDoc
from torch.utils.data import DataLoader


class Llmvdb:
    def __init__(
        self,
        embedding=None,
        llm=None,
        verbose: bool = False,
        file_path=None,
        workspace: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 3,
    ):
        self.embedding = embedding
        self.llm = llm
        self.verbose = verbose
        self.workspace = workspace
        self.file_path = file_path
        self.threshold = threshold
        self.top_k = top_k

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
            dataset, batch_size=32, shuffle=False, collate_fn=self.custom_collate_fn
        )
        doc_list = []

        for batch in tqdm(dataloader, desc="Processing dataset embedding"):
            texts = [data["text"] for data in batch if data["text"].strip() != ""]
            questions = [data["text"] for data in batch if data["text"].strip() != ""]
            context_embeddings = self.embedding.get_embedding(texts)
            question_embeddings = self.embedding.get_embedding(questions)

            for j, data in enumerate(batch):
                if data["text"].strip() != "":
                    doc_list.append(
                        ToyDoc(
                            text=data["text"],
                            context_embedding=context_embeddings[j],
                            question=data["question"],
                            question_embedding=question_embeddings[j],
                            tit_id=data["tit_id"],
                            ctx_id=data["ctx_id"],
                        )
                    )

        self.db.index(inputs=DocList[ToyDoc](doc_list))
        self.db.persist()

    def retrieve_document(self, prompt):
        query = ToyDoc(
            text=prompt, context_embedding=self.embedding.get_embedding(prompt)
        )
        search_parameters = {"search_field": "context_embedding"}
        results = self.db.search(
            inputs=DocList[ToyDoc]([query]), parameters=search_parameters, limit=5
        )

        input_document = ""
        over_threshold_indices = [
            idx for idx, value in enumerate(results[0].scores) if value > self.threshold
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
                    "#문서" + str(index) + "\n" + results[0].matches[index].text + "\n"
                )

        # 만약 threshold 0.8을 넘는게 있고 그 개수가 k개보다 많다면 top-k만 retrieve
        elif len(over_threshold_indices) >= self.top_k:
            for index in range(self.top_k):  # top-k (k=3)
                input_document += (
                    "#문서" + str(index) + "\n" + results[0].matches[index].text + "\n"
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

    def evaluate_model(self):
        dataset = CustomDataset(self.file_path)
        correct_count = 0
        total_queries = len(dataset) * 3

        for item in tqdm(dataset, desc=f"Evaluating {self.workspace}"):
            query_embedding = self.embedding.get_embedding(item["question"])
            query = item["question"]
            search_query = ToyDoc(question=query, context_embedding=query_embedding)
            search_parameters = {"search_field": "context_embedding"}

            search_results = self.db.search(
                inputs=DocList[ToyDoc]([search_query]),
                parameters=search_parameters,
                limit=3,
            )
            # print([i.hash_id for i in search_results[0].matches])
            count = sum(1 for m in search_results[0].matches if m.ctx_id == item["id"])
            correct_count += count
        accuracy = correct_count / total_queries if total_queries > 0 else 0
        print(f"Model: {self.workspace}, Accuracy: {accuracy}")
        return accuracy
