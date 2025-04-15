import os
import json
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

class GPT2JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data["prompt"] + "\n" + data["completion"]
                tokenized = tokenizer.encode(text, truncation=True, max_length=block_size)
                self.examples.append(torch.tensor(tokenized, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class GPT2DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer, block_size=128):
        self.examples = []
        half_size = block_size // 2
        for _, row in df.iterrows():
            prompt_tokens = tokenizer.encode(row["prompt"], truncation=True, max_length=half_size)
            completion_tokens = tokenizer.encode(row["completion"], truncation=True, max_length=half_size)
            input_ids = prompt_tokens + [tokenizer.eos_token_id] + completion_tokens
            input_ids = input_ids[:block_size]
            self.examples.append(torch.tensor(input_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return {'input_ids': self.examples[i]}

def pre_trained_model(df, output_dir, model_name="gpt2", num_train_epochs=3,
                              per_device_train_batch_size=2, block_size=128):
    """
    Fine-tunes GPT-2 using a DataFrame with 'prompt' and 'completion' columns.
    If a fine-tuned model exists in output_dir, training will continue from that model.
    """
    # Load from saved model if it exists, otherwise load the base model
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"Loading fine-tuned model from {output_dir}")
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
        model = GPT2LMHeadModel.from_pretrained(output_dir)
    else:
        print(f"Loading base model: {model_name}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Ensure tokenizer has a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Create dataset from DataFrame
    dataset = GPT2DataFrameDataset(df, tokenizer, block_size=block_size)

    # Data collator for language modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,  # Don't overwrite if continuing training
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":


    output_dir = "/Users/blairjdaniel/AI-Assistant-Springs/models/gpt2_finetuned"
    pre_trained_model(train_df, output_dir, num_train_epochs=3)