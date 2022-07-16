import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, pipeline
from datasets import Dataset
from datasets import load_metric
from transformers import TrainingArguments, Trainer

def encode(examples):
    return tokenizer(examples['text'], padding=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model_name = "cl-tohoku/bert-base-japanese-v2"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

df = pd.read_csv("annotation.csv", usecols= ['ラベル','文書'])
df = df.rename(columns={'ラベル': 'label', '文書': 'text'})
df['label'] = df['label'].replace({True: 1, False: 0})

dataset = Dataset.from_pandas(df)
dataset = dataset.map(encode)
train_test_data = dataset.train_test_split(test_size=0.2)
dev_data = train_test_data["test"].train_test_split(test_size=0.5)

training_args = TrainingArguments(output_dir="test_trainer")
metric = load_metric("accuracy")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=10)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_data["train"],
    eval_dataset=dev_data["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(eval_dataset=dev_data["train"])