import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.callbacks.base import BaseCallbackHandler
from llmvdb.embedding import HuggingFaceEmbedding, DPRTextEmbedding
from llmvdb.langchain import LangChain
import streamlit as st
from llmvdb import Llmvdb
from langchain.schema import ChatMessage


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def main(
    data_file_path: str,
    workspace: str,
    model_path: str,
    model_name: str = "klue/bert-base",
    is_dpr: bool = False,
    is_first: bool = False,
):
    if "llm" not in st.session_state:
        stream_handler = StreamHandler(st.empty())
        if is_dpr:
            embedding = DPRTextEmbedding("passage", model_path, model_name)
            question_embedding = DPRTextEmbedding("question", model_path, model_name)
        else:
            embedding = HuggingFaceEmbedding(model_name)

        llm = LangChain(callbacks=[stream_handler])
        st.session_state["llm"] = Llmvdb(
            embedding,
            llm,
            file_path=data_file_path,
            workspace=workspace,
            verbose=True,  # False로 설정시 터미널에 정보 출력 안됨
            threshold=0.0,  # threshold 값 조절 필요!
            top_k=5,
        )
        if is_first:
            st.session_state["llm"].initialize_db()  # vectordb저장, 처음에 한번만 실행

        # is_dpr = True 이면 question embedding으로 변경
        if is_dpr:
            st.session_state["llm"].change_embedding(question_embedding)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="안녕하세요! 저는 박물관 AI도슨트 슈팅스타 입니다.")
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = st.session_state["llm"].llm.set_callbacks([stream_handler])

            response = st.session_state["llm"].generate_response(prompt)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )


if __name__ == "__main__":
    main(
        # 질문-문서 데이터셋 경로
        data_file_path="data/train.jsonl",
        # 임베딩된 데이터가 저장되는(되어있는) 경로
        workspace="vectordb/museum_skt_kobert",
        # 학습된 dpr모델(.pth파일)의 경로
        model_path="data/museum_skt_kobert.pth",
        # 기반이 되는 모델
        model_name="skt/kobert-base-v1",
        # DPR 모델 사용 여부
        is_dpr=True,
        # 처음 실행 여부
        is_first=True,
        # 주의 : 이 값을 True로 하는 경우 = 모델을 바꾸거나, workspace를 변경했을때 True
        # 처음 폴더를 받은 상태에서 돌려보기만 할땐 False로 둬도 됨!
    )


# 실행 : 해당 경로에서 streamlit run main.py
