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
    is_dpr: bool = False,
    is_first: bool = False,
):
    if "llm" not in st.session_state:
        stream_handler = StreamHandler(st.empty())
        if is_dpr:
            embedding = DPRTextEmbedding("passage", model_path)
            question_embedding = DPRTextEmbedding("question", model_path)
        else:
            embedding = HuggingFaceEmbedding()

        llm = LangChain(callbacks=[stream_handler])
        st.session_state["llm"] = Llmvdb(
            embedding,
            llm,
            file_path=data_file_path,
            workspace=workspace,
            verbose=True,  # False로 설정시 터미널에 정보 출력 안됨
            threshold=0.1,
            top_k=3,
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
        data_file_path="data/dataset_ver1_1205.jsonl",  # 질문-문서 데이터셋 경로
        workspace="vectordb/kmrc_mrc_sample",  # 임베딩된 데이터가 저장되는(된) 경로
        model_path="data/kmrc_mrc.pth",  # 학습된 dpr모델(.pth파일)의 경로
        is_dpr=True,  # True -> model_path 에 학습된 모델 사용 / False -> model_name 사용
        is_first=False,  # True -> data_file_path의 문서 임베딩하고 저장함
        # False -> 저장된것 사용
        # 저장된것 사용할땐 False로 하고 workspace 경로 잘 지정해주면 됨
        # 주의 : is_first는 반드시 처음에만 True로 해야함, 데이터 덮어쓰기가 아닌 추가로 쌓아버림
    )

# 실행 : 해당 경로에서 streamlit run main.py
