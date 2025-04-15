# model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

def load_gpt_model(model_path):
    """
    Load the GPT model and tokenizer.

    Args:
        model_path (str): Path to the directory containing the model.

    Returns:
        tuple: (tokenizer, model) if successful, None otherwise.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading GPT model from {model_path}: {e}")
        return None