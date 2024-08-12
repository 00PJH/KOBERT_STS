import torch
from transformers import BertTokenizer, BertForSequenceClassification
from model_function import predict_label, classify_article

model_name = './STS_news_titles_KLUE_BERT_model'

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# 예시 기사 제목
article_title = '국내 주요 정보통신망 이상없어…IT 대란 예방사칭에 주의"(종합)'

# 예측
result = classify_article(article_title, model, tokenizer)

print(result)
