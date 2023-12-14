import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.callbacks.base import BaseCallbackHandler
from llmvdb.embedding import HuggingFaceEmbedding, DPRTextEmbedding
from llmvdb.langchain import LangChain
import streamlit as st
from llmvdb import Llmvdb
from langchain.schema import ChatMessage
import base64
from PIL import Image


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
    assistant_img = None,
    user_img = None
):
    
    
    st.title(":orange[국립중앙박물관] :blue[챗봇] :star2:")

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
            threshold=0.0, # threshold 값 조절 필요!
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
        if msg.role=="user":
            img = user_img
        else:
            img = assistant_img
        st.chat_message(msg.role, avatar=img).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user", avatar=user_img).write(prompt)

        with st.chat_message("assistant", avatar=assistant_img):
            stream_handler = StreamHandler(st.empty())
            llm = st.session_state["llm"].llm.set_callbacks([stream_handler])

            response = st.session_state["llm"].generate_response(prompt)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )

# 채팅 메시지의 배경을 하얀색으로 설정하는 CSS
chat_style = """
<style>
.stChatMessage {
    background-color: white !important;
    }
.sttitle {
    background-color: white !important;
}
</style>
"""


# # Streamlit에 CSS 적용
st.markdown(chat_style, unsafe_allow_html=True)

# 배경 설정하기
@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp{
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('../image/BGP_2.png')


if __name__ == "__main__":
    ai_path = "../image/AI_pic.png"
    user_path = "../image/user_cheunsik.jpg"


    main(
        # 질문-문서 데이터셋 경로
        data_file_path="data/train.jsonl",
        # 임베딩된 데이터가 저장되는(된) 경로
        workspace="vectordb/museum_skt_kobert",
        # 학습된 dpr모델(.pth파일)의 경로
        model_path="data/museum_skt_kobert.pth",
        # 기반이 되는 모델
        model_name = "skt/kobert-base-v1"
        # DPR 모델 사용 여부
        is_dpr=True,  # True -> model_path 에 학습된 모델 사용 / False -> model_name 사용
        # 처음 실행 여부
        is_first=False,  # True -> data_file_path의 문서 임베딩하고 저장함
        # False  -> 저장된것 사용
        # 주의 : 이 값을 True로 하는 경우 = 모델을 바꾸거나, workspace를 변경했을때 True
        #       저장된것 사용할땐 False로 하고 workspace 경로 잘 지정해주면 됨
        #       처음 폴더를 받은 상태에서 돌려보기만 할땐 False로 둬도 됨!
        
        # 주의 : is_first는 반드시 처음에만 True로 해야함, 데이터 덮어쓰기가 아닌 추가로 쌓아버림
        assistant_img=ai_path,
        user_img=user_path
    )


# 실행 : 해당 경로에서 streamlit run main.py
