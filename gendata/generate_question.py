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
            model='gpt-3.5-turbo-1106',
            max_retries=3
        ):

        # i=0
        dataset = []
        with jsonlines.open(self.jsonl_input_path) as file:
            for data in tqdm(file.iter()):
                try :
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
                                # 'content' : '너는 박물관 AI 도슨트야. 아래에 제시된 제목과 해설을 기반으로 관람객들이 물어볼만한 질문과 답변을 해설마다 10쌍씩 생성해줘. 문서에서 답변할 수 없는 질문은 제외하되, 직접적으로 정답이 되는 키워드 사용도 피해줘. 제목이 같다면 이전 문서를 참고해도 좋아. 포맷은 "질문: {q}, 답변: {a}" 으로 생성해줘. 질문에는 어떤 작품을 지칭하는지가 꼭 포함되어야해.'
                                'content' : '너는 박물관의 유물문서를 기반으로 검색하는 챗봇을 학습시킬 때 필요한 질문데이터를 만들어야해. 내가 제시한 문서를 먼저 요약한 뒤, 요약본을 기반으로 답변할 수 있는 질문을 관람객의 눈높이에 맞게 구체적으로 대신 물어봐주면 돼. 질문 생성은 철저히 제시된 문서만을 기반으로 해주고 질문에는 유물명이나 지칭하는 대상의 이름을 꼭 포함해야해. 답변 포맷은 question:내용\nanswer:내용 으로 부탁해.'
                            },
                            {
                                'role' : 'user',
                                'content' : f'문서의 해설:\n{description},\n\n 참고로 문서의 제목은[{title}]인데, 내용을 이해하는 데 참고만 하고 제목에 관련된 질문은 하지 마.\n\n 자, 이제 question과 answer 쌍을 생성해줘'
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
                except Exception as timeout:
                    dataset.append({'title':'실패', 'context':'실패', 'question':'실패'})
                    print(f'TimeOut, 전체 실패. 다음 Passage를 처리합니다. --> {timeout}')
                    continue

            # 일단 csv로도 저장
            df = pd.DataFrame(dataset)
            df.to_csv("./files/temp.csv", index=False, encoding="utf-8-sig")

        return dataset

    def query_double_check(
        self,
        model='gpt-3.5-turbo-1106',
        max_retries=3
        ):

        dataset = []
        with jsonlines.open(self.jsonl_input_path) as file:
            for data in tqdm(file.iter()):
                try :
                    ctx_id = data['ctx_id']
                    tit_ids = data['tit_ids']
                    answer = data['answer']
                    question = data['question']
                    context = data['context']
                    
                    # 시범용 (실제 체크할 때는 해제)
                    if tit_ids > 3 :
                        break

                    # 요청 보내기
                    response = openai.ChatCompletion.create(
                        model = model,
                        messages = [
                            {
                                'role' : 'system',
                                'content' : '너는 국립중앙박물관 소장품에 관한 문서를 검색하는 Retriever야. 문서(context)의 맥락과 내재된 의도를 파악하며 읽고, 내가 주는 question에 대한 답변이 context에 있으면 "T:답변 내용"을 말해주고, 답변을 찾을 수 없으면 "F:이유"를 출력하면 돼.'
                            },
                            {
                                'role' : 'user',
                                'content' : f'question:\n{question},\n\n context:{context}'
                            }
                        ]
                    )

                    # 재시도 3회까지 허용
                    retries = 0
                    while retries <= max_retries :
                        print(f'{retries}회 시도 중..')
                        if response.choices and response.choices[0].message['content'] :
                            try :
                                gen_answer = response.choices[0].message['content'].strip()
                                retries = max_retries+1
                            except Exception as e:
                                print(f"{e} T/F 답변 생성 실패 오류 발생, 재시도 중 ...")
                                retries += 1
                                if retries > max_retries:
                                    gen_answer = 'T/F 출력 후 실패함'
                                    continue
                        else :
                            retries += 1
                            if retries > max_retries:
                                print(f"재시도 횟수 초과. 다음 요청으로 넘어갑니다.")
                                gen_answer = '실패'
                                continue
                            print(f"연결 실패.. {retries}/{max_retries}회 에러")

                    dataset.append({'tit_ids':tit_ids, 'ctx_id':ctx_id, 't-f':gen_answer, 'origin_answer':answer, 'question':question, 'context':context})

                except Exception as timeout:
                    dataset.append({'title':'실패', 'context':'실패', 'question':'실패'})
                    print(f'TimeOut, 전체 실패. 다음 Passage를 처리합니다. --> {timeout}')
                    continue

            # 일단 csv로도 저장
            df = pd.DataFrame(dataset)
            df.to_csv("./files/temp_double_check.csv", index=False, encoding="utf-8-sig")
            df.to_json(self.jsonl_output_path, orient='records', lines=True, force_ascii=False)

        return dataset
    
    def save_data(self, dataset):
        with open(self.jsonl_output_path, 'w', encoding='utf-8') as write_file:
            write_file.write(json.dumps(dataset, ensure_ascii=False) + "\n")

if __name__ == "__main__" :
    gq = GenerateQuestion()
    dataset = gq.generate_question()
    gq.save_data(dataset)
    print('생성 완료.')
