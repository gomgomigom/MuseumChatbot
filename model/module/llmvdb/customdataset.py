import json
from tqdm.auto import tqdm


class CustomDataset:
    def __init__(self, file_path):
        self.documents_data = []
        # max_line = 500
        # lines = 0
        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file, desc="Reading file"):
                data = json.loads(line)
                text = f'{data.get("title", "")}\n{data.get("context", "")}{data.get("description","")}'
                question = data.get("question", "")
                tit_id = data.get("tit_id", "")
                ctx_id = data.get("ctx_id")
                self.documents_data.append(
                    {
                        "text": text,
                        "question": question,
                        "ctx_id": ctx_id,
                        "tit_id": tit_id,
                    }
                )
                # lines += 1
                # if lines >= max_line:
                #     break
