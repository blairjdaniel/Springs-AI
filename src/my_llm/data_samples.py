
import random
import pandas as pd

def generate_synthetic_friendly_data(n=1000):
    """
    Generates n synthetic examples with a prompt and a friendly completion.
    
    Returns:
        A pandas DataFrame with columns "prompt" and "completion".
    """
    # Define some prompt templates and their corresponding completion lists.
    prompt_templates = [
        ("I can come for a tour on {}.", "friendly_completions_two"),
        ("I'm ok with being put on a waitlist {}.", "friendly_completions_two"),
        ("Do you have more information about {}?", "friendly_completions_two"),
        ("I'm interested in learning more about {}.", "friendly_completions_two"),
        ("Can you tell me if {} is available?", "friendly_completions_two"),
        ("I'd like to inquire about {}.", "friendly_completions_two"),
        ("Do you have any availability {}?", "friendly_completions_two"),
        ("I can come on Friday, is that ok {}?", "friendly_completions_one"),
        ("I would like to speak to your manager {}.", "friendly_completions_three"),
        ("I missed your phone call yesterday {}.", "friendly_completions_one"),
        ("I am very upset with the customer service {}.", "friendly_completions_one"),
        ("Did you fix my problem {}?", "friendly_completions_one"),
        ("Thank you for your help {}.", "friendly_completions_three"),
        ("Are you open on Sunday {}?", "friendly_completions_three"),
        ("We appreciate your help {}.", "friendly_completions_three"),
        ("I left a review {}.", "friendly_completions_three")
    ]
    
    # Define the lists of friendly completions.
    friendly_completions_one = [
        "I see what you mean.",
        "I totally understand.",
        "I’m sorry to hear that you’re having trouble with this!",
        "We’re working on a solution for this.",
        "I’d feel the same way!",
        "I’ll let my team know about this!",   
    ]
    friendly_completions_two = [
        "Thanks for reaching out.",
        "Thanks for giving us a heads-up!",
        "I’m not sure, let’s find out!",
        "I just wanted to update you...",
        "I can absolutely help you with that!",
    ]
    friendly_completions_three = [
        "That’s a great question.",
        "We really appreciate you!",
        "That is super helpful!",
    ]
    
    # Map the completion list names to the actual lists.
    completion_lists = {
        "friendly_completions_one": friendly_completions_one,
        "friendly_completions_two": friendly_completions_two,
        "friendly_completions_three": friendly_completions_three
    }
    
    # Define additional details for filling placeholders in templates.
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    details = [
        "pricing details", "availability", "your services", "special promotions", "your tours"
    ]
    
    synthetic_data = []
    
    for _ in range(n):
        # Randomly select a template and its associated completion list.
        template, completion_list_name = random.choice(prompt_templates)
        completion_list = completion_lists[completion_list_name]
        
        # If the template requires a variable, pick one.
        if "{}" in template:
            if "tour" in template.lower():
                fill = random.choice(days)
            else:
                fill = random.choice(details)
            prompt_text = template.format(fill)
        else:
            prompt_text = template
        
        # Randomly select a friendly completion from the appropriate list.
        completion_text = random.choice(completion_list)
        
        synthetic_data.append({
            "prompt": prompt_text,
            "completion": completion_text
        })
    
    return pd.DataFrame(synthetic_data)


def generate_synthetic_friendly_data_json(n=1000, output_path=None):
    """
    Generates n synthetic examples and saves them as a JSONL file if output_path is provided.
    
    Returns:
        JSON string in JSONL format.
    """
    df = generate_synthetic_friendly_data(n)
    json_data = df.to_json(orient='records', lines=True)
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_data)
        print(f"Synthetic data saved to {output_path}")
    
    return json_data

# Example usage:
output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/cache/synthetic_data.jsonl"
json_output = generate_synthetic_friendly_data_json(1000, output_path)