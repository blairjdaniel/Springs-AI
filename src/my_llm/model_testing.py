import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

from sklearn.model_selection import train_test_split

# Load or define train_df before splitting
# Example: Loading train_df from a CSV file
train_df = pd.read_csv("/Users/blairjdaniel/AI-Assistant-Springs/data/google_colab/attitude_train.json")

# Split train_df into a smaller training set and a validation set
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

print("Updated Train DataFrame shape:", train_df.shape)
print("Validation DataFrame shape:", val_df.shape)

# Function to calculate BLEU score
def calculate_bleu(model, tokenizer, test_df):
    """
    Calculates the BLEU score for the model on the test dataset.
    """
    bleu_scores = []
    for _, row in test_df.iterrows():
        # Prepare the input prompt
        prompt = f"Category: {row['category']}\nIntent: {row['intent']}\nPrompt: {row['prompt']}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate the model's response
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Reference text (ground truth)
        reference = [row["completion"].split()]
        
        # Candidate text (model's output)
        candidate = generated_text.split()
        
        # Calculate BLEU score for this example
        bleu_score = sentence_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
    
    # Return the average BLEU score
    return sum(bleu_scores) / len(bleu_scores)



if __name__ == "__main__":
    # Example usage of the calculate_bleu function
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Assuming val_df is the test dataset
    bleu_score = calculate_bleu(model, tokenizer, val_df)
    print("Average BLEU score:", bleu_score)
  