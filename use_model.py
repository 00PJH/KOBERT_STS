import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from model_function import predict_label, classify_article

model_name = './STS_news_titles_KLUE_BERT_model'

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


df = pd.read_csv('NewsResult_20240812-20240812_only_titles.csv')
# 무작위로 샘플을 추출
sample_df = df.sample(n=100) 
count = 0


for title in sample_df["제목"] :
    result = classify_article(title, model, tokenizer)
    if result == 1 : 
        count += 1
        print(title) 
print(count)





# # 예시 기사 제목
# article_title = '2,580억 원 규모 ‘부산 미래성장 벤처펀드’ 조성'
# # 특전사·전방부대에 ‘로봇개’ 시범 배치

# # 예측
# result = classify_article(article_title, model, tokenizer)

# print(result)