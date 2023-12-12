from transformers import AutoTokenizer, AutoModel
import torch
from typing import Literal


class HuggingFaceEmbedding:
    def __init__(self, model_name: str = "klue/bert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, prompt):
        inputs = self.tokenizer(
            prompt, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            model_output = self.model(**inputs)
        sentence_embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        return sentence_embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class DPRTextEmbedding(HuggingFaceEmbedding):
    def __init__(
        self,
        mode: Literal["passage", "question"],
        model_path: str = "./data/kmrc_mrc.pth",
        model_name: str = "klue/bert-base",
    ):
        if mode not in ["passage", "question"]:
            raise ValueError("Mode must be 'passage' or 'question'")
        super().__init__(model_name)
        # super().__init__...은 아래 주석 두줄과 같음
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        self.mode = mode
        self.model_dict = {}
        self.load_model = torch.load(model_path, map_location=torch.device("cpu"))
        for key in self.load_model.keys():
            if key.startswith(f"module.{self.mode}_encoder."):
                self.model_dict[
                    key.replace(f"module.{self.mode}_encoder.", "")
                ] = self.load_model[key]
        self.model.load_state_dict(self.model_dict, strict=False)
