import logging
import traceback

logger = logging.getLogger("myapp")

def handle_error(error: Exception, context: str = ""):
    """
    Logs the error with traceback and context information.
    
    Parameters:
        error (Exception): The exception object caught.
        context (str): Optional additional context where the error occurred.
    """
    error_message = f"An error occurred in {context}: {str(error)}"
    # Log the error message and full traceback for debugging purposes
    logger.error(error_message)
    logger.error(traceback.format_exc())
    
    # Optionally, you can add additional error processing here
    # For example, sending alerts or performing clean-up actions.

    # Return the error message if needed
    return error_message

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the output directory and checkpoint directory
output_dir = "/Users/blairjdaniel/AI-Assistant-Springs/models/gpt2_finetuned"
checkpoint_dir = os.path.join(output_dir, "checkpoint-7500")

# Try to load the model and tokenizer from the checkpoint
try:
    if os.path.exists(checkpoint_dir):
        print(f"Loading model and tokenizer from checkpoint: {checkpoint_dir}")
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    else:
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' not found.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Falling back to the last saved fine-tuned model.")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
        model = GPT2LMHeadModel.from_pretrained(output_dir)
    except Exception as fallback_error:
        print(f"Error loading fine-tuned model: {fallback_error}")
        print("Exiting as no valid model could be loaded.")
        raise SystemExit

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

import os

def ensure_folder_exists(folder_path: str) -> None:
    """
    Ensures that a folder exists. If it doesn't, creates it.
    
    Parameters:
        folder_path (str): The path to the folder to check or create.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
    except Exception as e:
        print(f"Error creating folder '{folder_path}': {e}")
        raise

# Example usage:
if __name__ == "__main__":
    try:
        # Simulate code that may raise an exception
        1 / 0
    except Exception as e:
        handle_error(e, context="Main execution")

    # Test the model with a sample prompt
    sample_prompt = "What is the capital of Springs RV Resort?"
    input_ids = tokenizer.encode(sample_prompt, return_tensors="pt")
    try:
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        print("Generated Output:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    except Exception as test_error:
        print(f"Error during model testing: {test_error}")

    # Example usage: for checking for folder or creating a new one
    output_dir = "/Users/blairjdaniel/AI-Assistant-Springs/models/gpt2_finetuned"
    ensure_folder_exists(output_dir)