import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from llmvdb.embedding import HuggingFaceEmbedding, DPRTextEmbedding
from llmvdb import Llmvdb
import os
import matplotlib.pyplot as plt


def eval_model(model_paths: list, workspaces: list, is_first: bool = False):
    if not is_first:
        if os.path.exists("eval/model_performance.csv"):
            df = pd.read_csv("eval/model_performance.csv")
            existed_models = set("vectordb/" + df["model"])

    model_performance = []
    for model_path, workspace in zip(model_paths, workspaces):
        target_model = workspace.split("/")[-1]
        if not is_first and workspace in existed_models:
            print(f"{target_model}의 평가 결과는 이미 저장되어 있으므로 넘어갑니다.")
            continue
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

        accuracy = vdb.evaluate_model(target_model)

        accuracy["model"] = target_model
        model_performance.append(accuracy)
    df = pd.DataFrame(model_performance)

    return df


def make_graph(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    top_k_labels = [label for label in df.columns if label.startswith("Top_")]
    x_values = [int(label.split("_")[1]) for label in top_k_labels]

    for model in df["model"].unique():
        subset = df[df["model"] == model]
        y_values = subset[top_k_labels].iloc[0]
        plt.plot(x_values, y_values, label=model)

    plt.xlabel("Top_k")
    plt.ylabel("Accuracy")
    plt.title("Model-wise Top_k Accuracy")
    plt.legend()
    plt.xticks(x_values, top_k_labels, rotation=45)  # X축 눈금 설정
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 평가할 모델의 경로를 순서대로 적어줍니다 (workspace와 순서가 같아야함)
    # ""은 klue/bert-base 모델을 적용합니다 (dpr X)
    model_paths = [
        "",
        "data/museum_5epochs.pth",
        "data/merged_pn_5ep.pth",
        "data/museum_kdpr.pth",  # aihub_5epochs
    ]

    # 해당 모델로 임베딩한 index.bin이 있는 경로를 지정해줍니다.
    # 여기서 '/' 다음 부분이 csv파일에 모델이름으로 기록되게 됩니다.
    workspaces = [
        "vectordb/bert-base",
        "vectordb/museum_5epochs",
        "vectordb/merged_pn_5ep",
        "vectordb/aihub_5epochs",
    ]

    # eval/model_performance.csv 파일이 있다면 False로, False이면 이전에 했던 평가는 저장된값을 사용함
    is_first = False

    df = eval_model(model_paths, workspaces, is_first)
    if is_first:
        df.to_csv("eval/model_performance.csv", index=False, encoding="utf-8")
    else:
        df.to_csv(
            "eval/model_performance.csv",
            mode="a",
            index=False,
            header=False,
            encoding="utf-8-sig",
        )
    df = pd.read_csv("eval/model_performance.csv")
    print(df)
    make_graph(df)
