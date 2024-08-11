import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로드
df = pd.read_csv('concat_titles.csv')  # CSV 파일 이름 수정

# 데이터셋 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # 'labels'로 반환해야 손실을 계산할 수 있음
        }

# Tokenizer 설정
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

# Train/Test 데이터셋 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['title'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Dataset 객체 생성
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_len=128)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 모델 초기화
model = BertForSequenceClassification.from_pretrained('klue/bert-base', num_labels=2)
model.to(device)

# 평가 함수 정의
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# 트레이너 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 모델 훈련
trainer.train()

# 평가
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 최적화된 모델 저장
model.save_pretrained('./STS_titles_KLUE_BERT_model-2')
tokenizer.save_pretrained('./STS_titles_KLUE_BERT_model-2')

# Evaluation results: {'eval_loss': 0.11853419989347458,
#                       'eval_accuracy': 0.980634528224145,
#                         'eval_precision': 0.9812855980471928, 
#                         'eval_recall': 0.9804878048780488, 
#                         'eval_f1': 0.9808865392435949, 
#                         'eval_runtime': 10.7838, 
#                         'eval_samples_per_second': 225.06,
#                           'eval_steps_per_second': 14.095, 
#                           'epoch': 3.0}