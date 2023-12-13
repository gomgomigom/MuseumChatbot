import json
from tqdm.auto import tqdm
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, max_lines=None):
        self.documents_data = []
        seen_ctx_ids = set()
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                ctx_id = data.get("ctx_id")

                if ctx_id not in seen_ctx_ids:
                    self.documents_data.append(data)
                    seen_ctx_ids.add(ctx_id)

                if max_lines and len(self.documents_data) >= max_lines:
                    break

    def __len__(self):
        return len(self.documents_data)

    def __getitem__(self, idx):
        data = self.documents_data[idx]
        text = f'{data.get("title", "")}\n{data.get("context", "")}{data.get("description","")}'
        return {
            "text": text,
            "question": data.get("question", ""),
            "ctx_id": data.get("ctx_id"),
            "tit_id": data.get("tit_id"),
        }

    def get_all_data(self):
        all_data = []
        for data in self.documents_data:
            text = f'{data.get("title", "")}\n{data.get("context", "")}'
            item = {
                "text": text,
                "question": data.get("question", ""),
                "ctx_id": data.get("ctx_id"),
                "tit_id": data.get("tit_id"),
            }
            all_data.append(item)
        return all_data


class EvalCustomDataset(CustomDataset):
    def __init__(self, file_path):
        self.documents_data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                self.documents_data.append(data)
