from utils.helpers import extract_sender_name

def process_email_and_generate_response(email_text, sender_name, gpt_tokenizer, gpt_model):
    """
    Generate a tailored response using GPT based on the email text, sender, and category.
    """
    # Extract the sender's first name
    first_name = extract_sender_name(email_text)
    print(f"Debug: Extracted first name = {first_name}")

    # Construct the few-shot prompt
    few_shot_prompt = f"""
### Instructions ###
You are Kelsey, a friendly and professional personal assistant at Springs RV Resort. Your job is to respond to emails about tour date changes or cancellations in a polite and helpful manner. If the email does not mention a tour date change or cancellation, do not respond.

### Examples ###
Example 1:
From: Mary Smith
When would you like to come for a tour?: Hey, I'd like to reschedule my tour to Saturday at 11am.
Response: Hi Mary, I’ve updated your tour to Saturday at 11am. If you need to make further changes, please use this link to reschedule: https://calendly.com/springsrv/reschedule.

Example 2:
From: John Doe
When would you like to come for a tour?: Hi, I need to cancel my tour for next week. Can I reschedule for another time?
Response: Hi John, I’m sorry to hear you need to cancel your tour. Please use this link to reschedule at a time that works best for you: https://calendly.com/springsrv/reschedule.

### Task ###
From: {first_name}
{email_text}

Response:
"""

    # Tokenize the input and generate the output
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    inputs = gpt_tokenizer(few_shot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    outputs = gpt_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=gpt_tokenizer.pad_token_id
    )

    # Decode and return the generated response
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Debug: Raw generated response = {response}")
    if "Response:" in response:
        response = response.split("Response:")[0].strip()

    if any(greeting in response for greeting in ["Hi", "Dear", "Hey"]) and first_name in response:
        return {"responses": {}, "generated_response": response}
    else:
        return {"responses": {}, "generated_response": "No valid response generated."}