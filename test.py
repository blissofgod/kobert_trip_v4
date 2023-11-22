import torch
import torch.nn as nn
import random

from model.classifier import KoBERTforSequenceClassfication
from kobert_transformers import get_tokenizer


def load_wellness_answer(category_path="./data/trip_cose_mapping.txt",answer_path="./data/trip_cose_bot.txt"):

    #category_path = f"{root_path}/data/wellness_dialog_category.txt"

    #answer_path = f"{root_path}/data/wellness_dialog_answer.txt"


    c_f = open(category_path, 'r',encoding="utf-8")
    a_f = open(answer_path, 'r',encoding="utf-8")

    category_lines = c_f.readlines()
    answer_lines = a_f.readlines()

    category = {}
    answer = {}
    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        category[data[1][:-1]] = data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if (data[0] in keys):
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] = [data[1][:-1]]

    return category, answer


def kobert_input(tokenizer, str, device=None, max_seq_len=512):
    index_of_words = tokenizer.encode(str)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)

    # Padding Length
    padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
    }
    return data

def context(input_text,ouputtext,check_point_path="./checkpoint/kobert-wellness-text-classification.pth",category_path="./data/trip_cose_mapping.txt",answer_path="./data/trip_cose_bot.txt"):

    save_ckpt_path = check_point_path
    category, answer = load_wellness_answer(category_path,answer_path)

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    model = KoBERTforSequenceClassfication()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(ctx)
    model.eval()

    tokenizer = get_tokenizer()

    while 1:
        sent = input('\n'+ input_text)  # '요즘 기분이 우울한 느낌이에요'
        data = kobert_input(tokenizer, sent, device, 512)

        if ouputtext in sent:
            break
        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()

        print('category[str(max_index)]=', category[str(max_index)])
        answer_list = answer[category[str(max_index)]]
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        print('-' * 50)

        return category[str(max_index)]


def context_konlpy(input_text,ouputtext,check_point_path="./checkpoint/kobert-wellness-text-classification.pth",category_path="./data/trip_cose_mapping.txt",answer_path="./data/trip_cose_bot.txt"):

    save_ckpt_path = check_point_path
    category, answer = load_wellness_answer(category_path,answer_path)

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    model = KoBERTforSequenceClassfication()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(ctx)
    model.eval()

    tokenizer = get_tokenizer()





    while 1:

        sent = input_text  # '요즘 기분이 우울한 느낌이에요'
        data = kobert_input(tokenizer, sent, device, 512)

        if ouputtext in sent:
            break
        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()

        print('category[str(max_index)]=', category[str(max_index)])
        answer_list = answer[category[str(max_index)]]
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        print('-' * 50)
        if not "상관없음" in category[str(max_index)]:
            return category[str(max_index)]

if __name__ == "__main__":
    #context("inputtext,ouputext,checkpoint_path,category_path,answer_path)
    time=context("여행기간을 선택하세요","1박2일","./checkpoint/기간/kobert-period.pth",'./data/기간/trip_cose_mapping.txt','./data/기간/trip_cose_bot.txt')
    person = context("사람수를 입력하세요", "1박2일", "./checkpoint/person/kobert-person.pth",'./data/person/trip_cose_mapping.txt','./data/person/trip_cose_bot.txt' )
    cata = context_konlpy("테마를 입력하세요", "관계", "./checkpoint/cost/kobert-cost.pth",'./data/cost/trip_cose_mapping.txt', './data/cost/trip_cose_bot.txt')
    text=input('텍스트를 입력하시오')
    from konlpy.tag import Okt

    # Okt 객체 생성
    okt = Okt()

    # 텍스트 형태소 분석
    morphs = okt.morphs(text)
    print("형태소 분석 결과:", morphs)
    for i in morphs:
        cost = context_konlpy("테마를 입력하세요", "관계", "./checkpoint/cost/kobert-cost.pth",
                              './data/cost/trip_cose_mapping.txt', './data/cost/trip_cose_bot.txt')
        print("time=",cata)

    # print("time=",time)
    cost=input('여행 경비를 입력하시오')
    



