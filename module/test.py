from llmvdb.bm25 import CustomBM25

bm25 = CustomBM25(tokenizer="bert")
print(*bm25.get_top_n("인왕제색도 오른쪽엔 뭐가 있지?", 3), sep="\n\n")
print("=" * 25)
print(*bm25.get_top_n("청동 초두에 대해서 설명해줘", 3), sep="\n\n")
print("=" * 25)
print(*bm25.get_top_n("불교와 관련된 유물을 소개해줘", 3), sep="\n\n")
