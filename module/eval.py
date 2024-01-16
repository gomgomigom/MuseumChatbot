import pandas as pd
from tqdm.auto import tqdm
from llmvdb.embedding import HuggingFaceEmbedding, DPRTextEmbedding
from llmvdb import Llmvdb
import os
import matplotlib.pyplot as plt
from llmvdb.bm25 import CustomBM25
from llmvdb.customdataset import EvalCustomDataset
from tabulate import tabulate


def eval_model(
    model_paths: list,
    workspaces: list,
    model_names: list,
    is_first: bool = False,
):
    if not is_first:
        if os.path.exists("eval/model_performance.csv"):
            df = pd.read_csv("eval/model_performance.csv")
            existed_models = set("vectordb/" + df["model"])

    model_performance = []
    for model_path, workspace, model_name in zip(model_paths, workspaces, model_names):
        target_model = workspace.split("/")[-1]
        if not is_first and workspace in existed_models:
            print(f"{target_model}의 평가 결과는 이미 저장되어 있으므로 넘어갑니다.")
            continue
        print(model_path, workspace)
        if model_path:
            embedding = DPRTextEmbedding("question", model_path, model_name)
        else:
            print("====== klue/bert-base =====")
            embedding = HuggingFaceEmbedding(model_name)
        vdb = Llmvdb(
            embedding,
            workspace=workspace,
            verbose=True,
        )

        accuracy, accuracy_tit = vdb.evaluate_model(target_model)
        for accuracy_dict in [accuracy, accuracy_tit]:
            new_accuracy_dict = {
                f"Top_{k}": v for k, v in accuracy_dict.items() if isinstance(k, int)
            }
            for k in ["model", "question_len", "criteria"]:
                if k in accuracy_dict:
                    new_accuracy_dict[k] = accuracy_dict[k]
            model_performance.append(new_accuracy_dict)

    df = pd.DataFrame(model_performance)
    if is_first:
        df.to_csv(f"eval/model_performance.csv", index=False, encoding="utf-8")
    elif not df.empty:
        df.to_csv(
            f"eval/model_performance.csv",
            mode="a",
            index=False,
            header=False,
            encoding="utf-8",
        )
    return df


class GraphMaker:
    def __init__(self, df: pd.DataFrame):
        # df = df.drop(columns=["Top_50", "Top_20"])
        self.df = df
        self.top_k_labels = [label for label in df.columns if label.startswith("Top_")]
        self.x_values = [int(label.split("_")[1]) for label in self.top_k_labels]
        self.model_colors = {
            model: color
            for model, color in zip(
                df["model"].unique(), plt.rcParams["axes.prop_cycle"].by_key()["color"]
            )
        }

    def make_graph(self, criteria=None):
        plt.figure(figsize=(12, 8))

        if criteria:
            self.plot_criteria(criteria, linestyle="solid")
        else:
            self.plot_criteria("ctx_id", linestyle="solid")
            self.plot_criteria("tit_id", linestyle="dotted")
        plt.title(
            f"Model-wise Top_k Accuracy ({'ctx_id & tit_id ' if not criteria else criteria})"
        )
        self.finalize_plot()

    def plot_criteria(self, criteria, linestyle):
        for model in self.df["model"].unique():
            subset = self.df[
                (self.df["model"] == model) & (self.df["criteria"] == criteria)
            ]
            y_values = subset[self.top_k_labels].iloc[0]
            plt.plot(
                self.x_values,
                y_values,
                label=f"{model} ({criteria})",
                linestyle=linestyle,
                color=self.model_colors[model],
            )

    def finalize_plot(self):
        plt.xlabel("Top_k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.xticks(self.x_values, self.top_k_labels, rotation=45)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # 평가할 모델의 경로를 순서대로 적어줍니다 (workspace와 순서가 같아야함)
    # ""은 model_name모델을 적용합니다 (dpr X)
    model_paths = [
        "",  # klue/bert-base
        "data/museum_5epochs.pth",
        "data/merged_pn_5ep.pth",
        "data/museum_kdpr.pth",  # aihub_5epochs
        "data/museum_monologg_kobert.pth",  # 5ep
        "data/museum_skt_kobert.pth",
        "data/museum_skt_kobert_10ep.pth",
        "",  # skt
    ]

    # 해당 모델로 임베딩한 index.bin이 있는 경로를 지정해줍니다.
    # 여기서 '/' 다음 부분이 csv파일에 모델이름으로 기록되게 됩니다.
    workspaces = [
        "vectordb/bert-base",
        "vectordb/museum_bert-base-5ep",
        "vectordb/merged_pn_5ep",
        "vectordb/aihub_5epochs",
        "vectordb/museum_monologg_kobert_5ep",
        "vectordb/museum_skt_kobert_5ep",
        "vectordb/museum_skt_kobert_10ep",
        "vectordb/skt_kobert",
    ]

    # 기반 모델 이름(기본: klue/bert-base)
    model_names = [
        "klue/bert-base",
        "klue/bert-base",
        "klue/bert-base",
        "klue/bert-base",
        "monologg/kobert",
        "skt/kobert-base-v1",
        "skt/kobert-base-v1",
        "skt/kobert-base-v1",
    ]
    # eval/model_performance.csv 파일이 있다면 False로, False이면 이전에 했던 평가는 저장된값을 사용함
    is_first = False

    df = eval_model(model_paths, workspaces, model_names, is_first)

    df = pd.read_csv(f"eval/model_performance.csv")
    df = df[
        df["model"].isin(
            [
                # "museum_bert-base-5ep",
                # "museum_skt_kobert",
                # "aihub_5epochs",
                # "merged_pn_5ep",
                # "bm25_bert",
                "bm25_space",
                # "museum_monologg_kobert_5ep",
                # "bert-base",
                "skt_kobert",
                # "museum_skt_kobert_5ep",
                "museum_skt_kobert_10ep",
            ]
        )
    ]
    df = df[
        ["model", "criteria", "Top_1", "Top_2", "Top_3", "Top_5", "Top_7", "Top_10"]
    ]
    df_cols = [
        "model",
        "criteria",
        "Top_1",
        "Top_2",
        "Top_3",
        "Top_5",
        "Top_7",
        "Top_10",
    ]
    df_ctx_id = df[df["criteria"] == "ctx_id"]
    df_ctx_id = df_ctx_id[df_cols]
    df_tit_id = df[df["criteria"] == "tit_id"]
    df_tit_id = df_tit_id[df_cols]
    print(tabulate(df_ctx_id, headers="keys", tablefmt="grid", showindex=False))
    print()
    print(tabulate(df_tit_id, headers="keys", tablefmt="grid", showindex=False))
    print()
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    print()

    graph_maker = GraphMaker(df)
    graph_maker.make_graph("ctx_id")
    graph_maker.make_graph("tit_id")
    graph_maker.make_graph()
