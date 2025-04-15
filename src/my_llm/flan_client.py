from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import pandas as pd

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
train_encodings = tokenizer(list(df["prompt"]), truncation=True, padding=True, max_length=128)
labels = tokenizer(list(df["completion"]), truncation=True, padding=True, max_length=128)

# Prepare dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels["input_ids"][idx]),
        }

train_dataset = CustomDataset(train_encodings, labels)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Define training arguments
training_args = TrainingArguments(
    output_dir="/Users/blairjdaniel/AI-Assistant-Springs/models/flan-5",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()