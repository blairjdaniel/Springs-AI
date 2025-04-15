import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model  # Import LoRA for parameter-efficient fine-tuning


class GPT2DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer, block_size=128):
        self.examples = []
        half_size = block_size // 2
        for _, row in df.iterrows():
            category_tokens = tokenizer.encode(row["category"], truncation=True, max_length=half_size // 2)
            intent_tokens = tokenizer.encode(row["intent"], truncation=True, max_length=half_size // 2)
            prompt_tokens = tokenizer.encode(row["prompt"], truncation=True, max_length=half_size - len(category_tokens) - len(intent_tokens))
            prompt_tokens = category_tokens + intent_tokens + prompt_tokens
            completion_tokens = tokenizer.encode(row["completion"], truncation=True, max_length=half_size)
            input_ids = prompt_tokens + [tokenizer.eos_token_id] + completion_tokens
            input_ids = input_ids[:block_size]
            self.examples.append(torch.tensor(input_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return {'input_ids': self.examples[i]}


def train_gpt2_model_from_df(train_data, val_data, output_dir, model_name="gpt2", num_train_epochs=2,
                             per_device_train_batch_size=4, block_size=512):
    """
    Fine-tunes GPT-2 using pre-split train and validation DataFrames with 'prompt' and 'completion' columns.
    """

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Ensure the tokenizer has a pad token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Freeze lower layers of the model
    for param in model.transformer.h.parameters():
        param.requires_grad = False  # Freeze lower layers

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # Fine-tune attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Create custom datasets from the DataFrames
    train_dataset = GPT2DataFrameDataset(train_data, tokenizer, block_size=block_size)
    val_dataset = GPT2DataFrameDataset(val_data, tokenizer, block_size=block_size)

    # Data collator for language modeling (no masking for causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,  # Fewer epochs to avoid overfitting
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=5e-5,  # Small learning rate
        weight_decay=0.01,  # Add weight decay for regularization
        save_steps=500,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_steps=100,
        load_best_model_at_end=True,  # Load the best model based on validation loss
        metric_for_best_model="eval_loss",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset for evaluation
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model on the validation dataset
    print("Evaluating on validation dataset...")
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")

    # Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")