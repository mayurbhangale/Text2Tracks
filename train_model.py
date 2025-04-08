import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

df = pd.read_csv("training_data.csv")
dataset = Dataset.from_pandas(df)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def preprocess(example):
    x = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=16)
    y = tokenizer(example["target"], padding="max_length", truncation=True, max_length=8)
    x["labels"] = y["input_ids"]
    return x

tokenized = dataset.map(preprocess, remove_columns=["prompt", "target"])

training_args = TrainingArguments(
    output_dir="./text2tracks_model",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()

model.save_pretrained("text2tracks_model")
tokenizer.save_pretrained("text2tracks_model")
print("Model saved to text2tracks_model")