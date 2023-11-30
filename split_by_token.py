from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizerFast
import tiktoken
import json

class TokenSplitter :   
    def __init__(self, jsonl_input_path: str = "./files/museum_passage.jsonl", jsonl_output_path: str = "./files/passage_split.jsonl", encoding_name: str = "klue/bert-base") :
        '''
            Args:
                jsonl_input_path: 크롤링 데이터 원본을 담고 있는 JSONL 파일의 경로
                jsonl_output_path : 파일을 저장할 경로  
                encoding_name :
                    - BERT로 임베딩할 시 : "klue/bert-base"
                    - gpt-3.5-turbo 등으로 임베딩할 시 : "cl100k_base"
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path
        self.encoding_name = encoding_name

    def load_data(self):
        data_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(line)
        return data_list

    def bert_token_len(self, text):
        encoding_name = self.encoding_name
        tokenizer = BertTokenizerFast.from_pretrained(encoding_name)
        tokens = tokenizer(text, return_tensors='pt')
        return len(tokens)

    def bert_split(self, data):
        '''
            Desc:
                원본 크롤링 파일 (.jsonl) 문장에서 설명 부분이 380토큰 (BERT 기준) 넘어갈 경우 Split함
            Args:
                data : 크롤링 파일
                jsonl_output_path : 파일을 저장할 경로  
        '''
        encoding_name = self.encoding_name
        tokenizer = BertTokenizerFast.from_pretrained(encoding_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(chunk_size=380, chunk_overlap=50, tokenizer=tokenizer, separators=["한편", "이 밖에도", ". "], keep_separator=False)
        
        # TODO: key가 description이어야 함 -> 없으면 KeyError 발생
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file :
            for line in data:
                try :
                    text_dict = json.loads(line)
                    text = text_dict['description']
                    # print(text)
                except KeyError:
                    raise KeyError("작품 설명의 key값을 'description'으로 설정하세요.")

                split_data = splitter.split_text(text)
                for single_split_data in split_data :
                    # print(f'split... -> {single_split_data}')
                    new_line = text_dict
                    new_line['description'] = single_split_data
                    print(f'new_line: {new_line}')
                    json.dump(new_line, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')

    def tiktoken_len(self, text, encoding_name="cl100k_base"):
        '''
            Desc:
                tiktoken library를 활용해 토큰 길이를 계산하는 함수
            Args:
                1. text : 토큰 길이 측정할 텍스트
                2. encoding_name : 토큰으로 변환할 방식 지정
                    - cl100k_base : gpt-3.5-turbo, gpt-4, text-embedding-ada-002 모델 사용시
                    - p50k_base : text-davinci-002, text-davinci-003 모델 사용시
                    - r50k_base (or gpt2) : GPT-3 models like "davinci"
        '''
        tokenizer = tiktoken.get_encoding(encoding_name)
        # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # 위 코드로 안될 시 모델 이름 넣어서 이걸로 해보기
        tokens = tokenizer.encode(text)
        return len(tokens)

    def tiktoken_split(self, data):
        '''
            Desc:
                원본 크롤링 파일 (.jsonl) 문장에서 설명이 2000토큰이 (cl100k_base 기준) 넘어갈 경우 Split함
            Args:
                data : 크롤링 파일
        '''
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)

        with open(self.jsonl_output_path, 'w', encoding='utf-8') as jsonl_file :
            for line in data:
                # description 부분만 받아오기 -> 없으면 KeyError 발생
                try :
                    text_dict = json.loads(line)
                    text = text_dict['description']
                except KeyError:
                    raise KeyError("작품 설명의 key값을 'description'으로 설정하세요.")
                token_len = self.tiktoken_len(text)

                # 토큰 길이가 2000을 넘을 경우 문서 Split
                if token_len > 2000 :
                    split_data = splitter.split_text(text)
                    for single_split_data in split_data :
                        # print(f'split... -> {single_split_data}')
                        new_line = text_dict
                        new_line['description'] = single_split_data
                        json.dump(new_line, jsonl_file, ensure_ascii=False)
                        jsonl_file.write('\n')
                else :
                    json.dump(text_dict, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
    
    def tiktoken_len_check(self, encoding_name="cl100k_base"):
        len_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokenizer = tiktoken.get_encoding(encoding_name)
                tokens = tokenizer.encode(line)
                line = json.loads(line)
                len_list.append({len(tokens):line['title']})
        print(len_list)
        return len(tokens)
    
    def bert_len_check(self, encoding_name):
        encoding_name = self.encoding_name
        len_list = []
        over_list = []
        with open(self.jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokenizer = BertTokenizerFast.from_pretrained(encoding_name)
                tokens = tokenizer(line, return_tensors='pt')['input_ids']
                line = json.loads(line)
                if tokens.size(1) > 450 :
                    len_list.append({line['title']:tokens.size(1)})
                print(f'"{line["title"]}" 하는 중.. Token Length : {tokens.size(1)}')
        print(len_list)
        return tokens.size(1)

    def process(self):
        if self.encoding_name == "klue/bert-base" :
            print('Running bert_split')
            data = self.load_data()
            # print(data[1])
            self.bert_split(data)
        elif self.encoding_name == "cl100k_base" :
            print('Running tiktoken_split')
            data = self.load_data()
            self.tiktoken_split(data)

if __name__ == "__main__" : # python preprocessing.py로 실행할 경우
    
    ## bert 기반 tokenizing 후 split을 진행할 경우
    ## museum_passage.jsonl : 원본 크롤링 데이터, passage_bert_split.jsonl : split 후 저장할 파일명 (변경 가능)
    # pp = Preprocessing('./files/museum_passage.jsonl', './files/passage_bert_split.jsonl', encoding_name='klue/bert-base')
    # pp.process()

    ## tiktoken(cl100k-base)으로 tokenizing 후 split을 진행할 경우 (gpt-3.5-turbo)
    ## museum_passage.jsonl : 원본 크롤링 데이터, passage_tiktoken_split.jsonl : split 후 저장할 파일명 (변경 가능)
    # pp = Preprocessing('./files/museum_passage.jsonl', './files/passage_tiktoken_split.jsonl', encoding_name='cl100k_base')
    # pp.process()

    ## tiktoken 기반 토큰 갯수 체크
    ## input만 넣어도 됨
    # pp = Preprocessing(jsonl_input_path='./files/passage_tiktoken_split.jsonl', jsonl_output_path='./files/passage_tiktoken_split.jsonl')
    # pp.tiktoken_len_check()

    ## bert 기반 토큰 갯수 체크
    ## input만 넣어도 됨
    # pp = Preprocessing(jsonl_input_path='./files/passage_bert_split.jsonl', jsonl_output_path='./files/passage_bert_split.jsonl')
    # pp.bert_len_check()

    pp = Preprocessing(jsonl_input_path='./files/title_desc_passage.jsonl')
    pp.bert_len_check(encoding_name='klue/bert-base')