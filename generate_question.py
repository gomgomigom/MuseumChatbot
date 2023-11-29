# TODO: .env에 {OPENAI_API_KEY=...} 형식의 openai key가 필요함
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import pandas as pd
import jsonlines
import json

import openai

class GenerateQuestion :
    def __init__(self, jsonl_input_path: str = "./files/title_desc_passage.jsonl", jsonl_output_path: str = "./files/qa_gpt_dataset.jsonl", title: str = "title", description: str = "description") :
        '''
            Args:
                jsonl_input_path : 질문을 생성할 기반 문서 (JSONL) 파일의 경로, 430 토큰 미만으로 Split 할 것을 권장.
                json_output_path : Q-A-CTXs DPR 데이터셋을 저장할 경로 (JSON)

                title : 데이터 내 "제목"의 필드값 (key)
                description : 데이터 내 "큐레이션 설명"의 필드값 (key)
        '''
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_output_path = jsonl_output_path

        self.title = title
        self.description = description

    def load_data(self):
        data_list = []
        with jsonlines.open(self.jsonl_input_path) as file:
            for line in file.iter():
                data_list.append(line)
        return data_list
    
    def generate_question(
            self,
            model='gpt-3.5-turbo',
            max_retries=3
        ):

        # i=0
        dataset = []
        with jsonlines.open(self.jsonl_input_path) as file:
            try :
                for data in tqdm(file.iter()):
                    title = data[self.title]
                    # print(f'load data.. title:{title}')
                    description = data[self.description]
                    # print(f'load data.. description:{description}')
                    
                    # 요청 보내기
                    response = openai.ChatCompletion.create(
                        model = model,
                        messages = [
                            {
                                'role' : 'system',
                                'content' : '너는 박물관 AI 도슨트야. 아래에 제시된 제목과 해설을 기반으로 관람객들이 물어볼만한 질문과 답변을 해설마다 10쌍씩 생성해줘. 문서에서 답변할 수 없는 질문은 제외하되, 직접적으로 정답이 되는 키워드 사용도 피해줘. 제목이 같다면 이전 문서를 참고해도 좋아. 포맷은 "질문: {q}, 답변: {a}" 으로 생성해줘. 질문에는 어떤 작품을 지칭하는지가 꼭 포함되어야해.'
                            },
                            {
                                'role' : 'user',
                                'content' : f'문서->\n제목: {title} \n해설:{description}]'
                            }
                        ]
                    )

                    # 재시도 3회까지 허용
                    retries = 0
                    while retries <= max_retries :
                        print(f'{retries}회 시도 중..')
                        if response.choices and response.choices[0].message['content'] :
                            try :
                                qa_pair = response.choices[0].message['content'].strip()
                                retries = max_retries+1
                            except Exception as e:
                                print(f"{e} question 생성 실패 오류 발생, 재시도 중 ...")
                                retries += 1
                                if retries > max_retries:
                                    qa_pair = '생성 후 실패'
                                    continue
                        else :
                            retries += 1
                            if retries > max_retries:
                                print(f"재시도 횟수 초과. 다음 요청으로 넘어갑니다.")
                                qa_pair = '실패'
                                continue
                            print(f"연결 실패.. {retries}/{max_retries}회 에러")

                    dataset.append({'title':title, 'context':description, 'question':qa_pair})
                    # 시범용
                    # if i == 10 :
                        # break
                    # i += 1
            except Exception as ee:
                dataset.append({'title':'실패', 'context':'실패', 'question':'실패'})
            
            # 일단 csv로도 저장
            df = pd.DataFrame(dataset)
            df.to_csv("./files/qa_gpt_df.csv", index=False, encoding="utf-8-sig")

        return dataset
    
    def save_data(self, dataset):
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as write_file:
            write_file.write(json.dumps(dataset, ensure_ascii=False) + "\n")

if __name__ == "__main__" :
    gq = GenerateQuestion()
    dataset = gq.generate_question()
    gq.save_data(dataset)
    print('끝.')
