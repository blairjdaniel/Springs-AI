from utils.helpers import extract_sender_name
from my_prompt_engineering.few_shot import generate_few_shot_prompt

def generate_response(email_text, sender, email_category, gpt_tokenizer, gpt_model):
    """
    Generate a tailored response using GPT based on the email text, sender, and category.

    Args:
        email_text (str): The body of the email.
        sender (str): The sender's name or email address.
        email_category (str): The category of the email.
        gpt_tokenizer: The tokenizer for the GPT model.
        gpt_model: The GPT model.

    Returns:
        str: The generated response.
    """
    first_name = extract_sender_name(email_text)
    cancellation_keywords = ["cancel", "cancellation", "reschedule", "change"]
    email_lower = email_text.lower()

    extracted_details = []
    if any(keyword in email_lower for keyword in cancellation_keywords):
        extracted_details.append("The email mentions a cancellation or rescheduling request.")
    if "tour" in email_lower:
        extracted_details.append("The email mentions a tour request.")

    few_shot_prompt = generate_few_shot_prompt(email_text, first_name, email_category)
    dynamic_instruction = (
        "Please generate a response that incorporates the following details extracted from the email:\n"
        + '\n'.join(extracted_details) + "\n\n"
        if extracted_details else ""
    )

    prompt = (
        f"Category: {email_category.capitalize()}\n\n"
        f"{few_shot_prompt}\n\n"
        "###\n"
        f"{dynamic_instruction}"
        "Do not include unrelated information or hallucinate details:\n"
    )

    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    max_input_length = 1024 - 150
    inputs = gpt_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)

    outputs = gpt_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=gpt_tokenizer.pad_token_id
    )

    if outputs is None or len(outputs) == 0:
        raise ValueError("No output tokens generated. Check your generation parameters.")

    generated_response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "###" in generated_response:
        generated_response = generated_response.split("###")[0].strip()

    if generated_response.strip() == few_shot_prompt.strip():
        raise ValueError("The model copied the few-shot examples instead of generating a tailored response.")

    return generated_response