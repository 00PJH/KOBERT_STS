import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = './STS_news_titles_KLUE_BERT_model'

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

def predict_label(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

def classify_article(title, model, tokenizer):
    label = predict_label(title, model, tokenizer)
    return "국방 관련 기사입니다." if label == 1 else "국방 관련 기사가 아닙니다."

# 예시 기사 제목
article_title = '국내 주요 정보통신망 이상없어…IT 대란 예방사칭에 주의"(종합)'

# 예측
result = classify_article(article_title, model, tokenizer)
print(result)
