from llmvdb.bm25 import CustomBM25
from llmvdb.customdataset import EvalCustomDataset
from tqdm.auto import tqdm
import pandas as pd

conditions = [
    ("bm25_bert", "ctx_id"),
    ("bm25_bert", "tit_id"),
    ("bm25_space", "ctx_id"),
    ("bm25_space", "tit_id"),
]
results = []
for model, criteria in tqdm(conditions, desc="Processing conditions"):
    row = {}
    row["model"] = model
    row["criteria"] = criteria
    bm25 = CustomBM25(tokenizer=model.split("_")[1])

    test_dataset = EvalCustomDataset("data/test.jsonl").get_all_data()

    for k in tqdm([1, 2, 3, 5, 7, 10, 20, 50], desc=f"Accuracy 측정중"):
        accuracy = bm25.calculate_accuracy(test_dataset, k, id_key=criteria)
        row[f"Top_{k}"] = accuracy
    results.append(row)

df_results = pd.DataFrame(results)
df_origin = pd.read_csv("eval/model_performance.csv")
df_combined = pd.concat([df_origin, df_results], axis=0)
df_combined.to_csv("eval/model_performance.csv", index=False, encoding="utf-8")
