import json
from tqdm.auto import tqdm
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, max_lines=None):
        self.documents_data = []
        with open(file_path, "r", encoding="utf-8") as file:
            if max_lines:
                for line, _ in zip(file, range(max_lines)):
                    data = json.loads(line)
                    self.documents_data.append(data)
            else:
                for line in file:
                    data = json.loads(line)
                    self.documents_data.append(data)

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
