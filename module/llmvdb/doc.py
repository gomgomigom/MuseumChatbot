from docarray import BaseDoc
from docarray.typing import NdArray
from typing import Optional


class ToyDoc(BaseDoc):
    text: str = ""
    context_embedding: Optional[NdArray[768]]
    question: str = ""
    question_embedding: Optional[NdArray[768]]
    tit_id: str = ""
    ctx_id: str = ""
