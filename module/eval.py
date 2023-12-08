import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from llmvdb.embedding import HuggingFaceEmbedding, DPRTextEmbedding
from llmvdb import Llmvdb


def eval_model(model_paths, workspaces):
    model_performance = []
    for model_path, workspace in zip(model_paths, workspaces):
        print(model_path, workspace)
        if model_path:
            embedding = DPRTextEmbedding("question", model_path)
        else:
            print("====== klue/bert-base =====")
            embedding = HuggingFaceEmbedding()
        vdb = Llmvdb(
            embedding,
            workspace=workspace,
            verbose=True,
        )

        target_model = workspace.split("/")[-1]
        accuracy = vdb.evaluate_model(target_model)

        accuracy["model"] = target_model
        model_performance.append(accuracy)

    df = pd.DataFrame(model_performance)

    df.to_csv("eval/model_performance.csv", index=False, encoding="utf-8-sig")
    print(df)


if __name__ == "__main__":
    # 평가할 모델의 경로를 순서대로 적어줍니다 (workspace와 순서가 같아야함)
    # ""은 klue/bert-base 모델을 적용합니다 (dpr X)
    model_paths = ["data/museum_5epochs.pth", ""]

    # 해당 모델로 임베딩한 index.bin이 있는 경로를 지정해줍니다.
    # 여기서 '/' 다음 부분이 csv파일에 모델이름으로 기록되게 됩니다.
    workspaces = ["vectordb/museum_5epochs", "vectordb/bert-base"]
    eval_model(model_paths, workspaces)
