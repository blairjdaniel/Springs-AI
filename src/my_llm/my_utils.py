import re
import yaml
import string
import json

def load_yaml_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def clean_text(text: str, lower_case: bool = True, remove_punctuation: bool = True) -> str:
    if lower_case:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

def count_words(text: str) -> int:
    return len(text.split())

def extract_emails(text: str) -> list:
    pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    return re.findall(pattern, text)

def format_phone_number(phone: str) -> str:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

def format_phone_numbers_in_text(text: str) -> str:
    """
    Searches the text for phone number patterns and replaces them with a formatted version.
    The regex pattern below matches numbers like 604-557-6168, 6045576168, or 604.557.6168.
    Adjust the pattern as necessary.
    """
    phone_pattern = re.compile(r'\b(\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4})\b')
    matches = phone_pattern.findall(text)
    for match in matches:
        formatted = format_phone_number(match)
        text = text.replace(match, formatted)
    return text

def clean_text_for_bert(text: str, cleaning_conf: dict) -> str:
    """
    Cleans text for training BERT based on cleaning configuration.
    
    Parameters:
        text (str): Input text to clean.
        cleaning_conf (dict): Cleaning configuration (e.g., lower_case, remove_punctuation).
        
    Returns:
        str: Cleaned text.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove specific unwanted characters/substrings
    text = text.replace("**", "")
    text = text.replace("#", "")
    text = text.replace("-", "")
    
    # Apply lower-casing if enabled
    if cleaning_conf.get("lower_case", False):
        text = text.lower()
    
    # Remove punctuation if enabled
    if cleaning_conf.get("remove_punctuation", False):
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Format phone numbers found in the text
    text = format_phone_numbers_in_text(text)
    
    # Optionally, add any additional cleaning (like removing stopwords) here.
    
    return text

# Sample usage for testing (can be removed or placed under '__main__')
if __name__ == "__main__":
    sample_text = "Hello, World! This is a sample text. Contact us at example@test.com. Call us at 604-557-6168."
    cleaning_conf = {
        "lower_case": True,
        "remove_punctuation": False  # Set False so demo phone number isn't stripped before formatting
    }
    print("Clean Text for BERT:", clean_text_for_bert(sample_text, cleaning_conf))


def convert_insta_txt_to_jsonl(input_path: str, output_path: str):
    """
    Reads a file with one JSON object per line (insta.txt) and writes a new JSON Lines file.
    
    Each line should be a valid JSON string which will be re-serialized into output_path.
    """
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                # Try to parse each line as JSON
                data = json.loads(line)
                # Write the JSON object as a single line in the output file
                outfile.write(json.dumps(data) + "\n")
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)

def remove_empty_body_entries(input_path: str, output_path: str) -> None:
    """
    Reads a JSON file containing a list of entries and writes a new JSON file 
    with any entry removed where the "body" key is an empty string.
    
    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Filter entries: remove those where "body" is exactly ""
    filtered_data = [entry for entry in data if entry.get("body", None) != ""]
    
    with open(output_path, "w") as outfile:
        json.dump(filtered_data, outfile, indent=4)

def add_form_field(input_path: str, output_path: str, form_type: str) -> None:
    """
    Reads a JSON file containing a list of entries, adds a "form" key with the provided form_type
    to each entry, and writes the updated list to a new JSON file.
    
    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
        form_type (str): The form type to add (e.g., "waitlist", "contact", "phase3_sales").
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Add the "form" field to each entry
    for entry in data:
        entry["form"] = form_type
    
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def add_company_voice(input_path: str, output_path: str, company_voice: str) -> None:
    """
    Reads a JSONL file, adds a "company_voice" key with the given value to each entry,
    and writes the modified entries to a new JSONL file.
    
    Parameters:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSONL file.
        company_voice (str): The metadata value for the company's voice.
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Add the "company_voice" field to each entry
    for entry in data:
        entry["company_voice"] = company_voice
    
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def load_forms_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def enrich_email_with_contact_response(email_entry: dict, forms_config: dict) -> dict:
    form_type = email_entry.get("form", "").lower()
    baseline = {}
    forms = forms_config.get("forms", {})
    
     # Create a lower-cased key mapping for forms
    lower_forms = { key.lower(): value for key, value in forms.items() }
    
    # Retrieve the corresponding baseline response (if any)
    email_entry["baseline_response"] = lower_forms.get(form_type, lower_forms.get("contact", {}))
    return email_entry



def process_emails(emails_jsonl_path: str, forms_yaml_path: str, output_path: str) -> None:
    forms_config = load_forms_config(forms_yaml_path)
    enriched_entries = []
    with open(emails_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                enriched_entry = enrich_email_with_contact_response(entry, forms_config)
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
    # Write the enriched emails back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")

import json
import yaml

def load_socials_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def enrich_instagram_post(instagram_entry: dict, socials_config: dict) -> dict:
    # Retrieve the instagram template from the socials config
    instagram_template = socials_config.get("socials", {}).get("instagram", {})
    
    # Get the response template string
    response_template = instagram_template.get("response_template", "")
    
    # Format the template if needed by inserting the post content.
    # For instance, if the response_template uses the placeholder {post_content}, fill it with the actual text.
    if "{post_content}" in response_template:
        formatted_response = response_template.format(post_content=instagram_entry.get("text", ""))
    else:
        formatted_response = response_template
        
    # Save the formatted response into the entry under baseline_response
    instagram_entry["baseline_response"] = formatted_response

    # Optionally, if you want to include additional data such as guidelines or examples, you can add those too.
    instagram_entry["social_guidelines"] = instagram_template.get("guidelines", [])
    instagram_entry["social_examples"] = instagram_template.get("examples", [])
    
    return instagram_entry

def process_instagram_posts(instagram_jsonl_path: str, socials_yaml_path: str, output_path: str) -> None:
    socials_config = load_socials_config(socials_yaml_path)
    enriched_entries = []
    with open(instagram_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Ensure the entry is for instagram by checking its type (or other key)
                if entry.get("type", "").lower() == "instagram":
                    enriched_entry = enrich_instagram_post(entry, socials_config)
                else:
                    enriched_entry = entry
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
    
    # Write the enriched entries back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")

def enrich_website_content(website_entry: dict, socials_config: dict) -> dict:
    # Retrieve the website template from the socials config
    website_template = socials_config.get("socials", {}).get("website", {})
    
    # The website template's response_template expects a placeholder {homepage_content}
    response_template = website_template.get("response_template", "")
    
    # Format the response by inserting the website entry's content into the template
    formatted_response = response_template.format(
        homepage_content=website_entry.get("content", "")
    )
    
    # Add the baseline response (you can also include subject if needed)
    website_entry["baseline_response"] = {
        "subject": website_template.get("label", "Website Content"),
        "body": formatted_response
    }
    
    # Optionally, include guidelines and examples from the configuration
    website_entry["guidelines"] = website_template.get("guidelines", [])
    website_entry["examples"] = website_template.get("examples", [])
    
    return website_entry

def process_website_posts(website_jsonl_path: str, socials_yaml_path: str, output_path: str) -> None:
    socials_config = load_socials_config(socials_yaml_path)
    enriched_entries = []
    with open(website_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Check that this entry is marked as website content
                if entry.get("type", "").lower() == "website":
                    enriched_entry = enrich_website_content(entry, socials_config)
                else:
                    enriched_entry = entry
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
                
    # Write the enriched entries back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")


import re
import json

def parse_brandbook(text):
    # Use regex to capture each numbered section.
    # This regex expects a number followed by a period and a space.
    pattern = r"\s*(\d+)\.\s+"
    # Find all split positions
    splits = re.split(pattern, text.strip())
    
    # The first element might be the header (if present), so we ignore if it doesn't look like a number.
    # The splits result will be like: [header?, num, content, num, content, ...]
    entries = {}
    if len(splits) >= 3:
        # If the first element is not a number, ignore it and process subsequent number-content pairs.
        i = 1 if not splits[0].strip().isdigit() else 0  
        # Adjust starting index: if splits[0] is header, then i = 1
        if i == 1:
            header = splits[0].strip()
            entries["header"] = header
        # Process the remaining parts in pairs: number then content.
        for j in range(i, len(splits) - 1, 2):
            number = splits[j].strip()
            content = splits[j+1].strip()
            entries[number] = content
    
    return entries

def convert_brandbook_to_examples(brandbook_text):
    """
    Converts a brandbook text file into a list of training examples.
    Each example will have a "prompt" and a "completion" field.
    
    The prompt is built from the section number (and header if available) and
    the completion is the corresponding content.
    
    Parameters:
        brandbook_text (str): The raw brandbook text.
    Returns:
        list: List of example dictionaries with keys "prompt" and "completion".
    """
    parsed = parse_brandbook(brandbook_text)
    examples = []
    header = parsed.pop("header", "")
    for number, content in parsed.items():
        prompt = f"Brandbook Section {number}"
        if header:
            prompt = f"{header}\nSection {number}:"  # include header if wanted
        completion = content
        examples.append({"prompt": prompt, "completion": completion})
    return examples

import json

def convert_brandbook_file(input_path: str, output_path: str, default_prompt: str = "Brandbook entry:") -> None:
    """
    Reads a JSONL file containing brandbook entries and converts each entry to have "prompt"
    and "completion" fields.
    
    - If an entry already has "prompt" and "completion", it is written as is.
    - Otherwise, if the entry has a "text" field, it is converted to:
          {"prompt": default_prompt, "completion": entry["text"]}
    
    Parameters:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path where the converted JSONL file will be written.
        default_prompt (str): Prompt to use for entries that only contain "text".
    """
    examples = []
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "prompt" in entry and "completion" in entry:
                new_entry = entry
            elif "text" in entry:
                new_entry = {"prompt": default_prompt, "completion": entry["text"]}
            else:
                # Skip entries that do not have expected keys
                continue
            outfile.write(json.dumps(new_entry) + "\n")
            examples.append(new_entry)
    print(f"Converted {len(examples)} entries into prompt/completion format. Output written to {output_path}")
    

import glob
import json
import os
import pandas as pd

def load_all_jsonl_files(directory: str) -> list:
    """
    Loads and combines data from all JSONL files in the specified directory.
    
    Parameters:
        directory (str): Path to the directory containing JSONL files.
    
    Returns:
        A list with the combined data entries from all processed JSONL files.
    """
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
    return all_data


def load_normalized_jsonl(file_path: str) -> pd.DataFrame:
    """
    Loads a JSONL file into a Pandas DataFrame and normalizes nested fields.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        pd.DataFrame: A normalized DataFrame containing the data.
    """
    
    # Read the JSONL file using Pandas
    df = pd.read_json(file_path, lines=True)
    # Normalize the DataFrame entries
    normalized_df = pd.json_normalize(pd.DataFrame(df).to_dict(orient='records'))
    return normalized_df

def load_txt_files_to_df(directory: str) -> pd.DataFrame:
    """
    Loads all .txt files in a directory into a pandas DataFrame.
    
    Parameters:
        directory (str): Path to the directory containing .txt files.
        
    Returns:
        pd.DataFrame: A DataFrame with columns "filename" and "content".
    """
    data = []
    # List all files in the directory
    for fname in os.listdir(directory):
        if fname.lower().endswith(".txt"):
            file_path = os.path.join(directory, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                data.append({"filename": fname, "content": content})
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return pd.DataFrame(data)

from sklearn.model_selection import train_test_split
import pandas as pd

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Splits a DataFrame into train and test DataFrames.

    Parameters:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Fraction of the data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

# Mapping from "form" value to a baseline response.
BASELINE_RESPONSE = {
    "contact": "Hi {first_name}, Thanks so much for reaching out! My name is Kelsey, and I’m the Sales Assistant here at Springs RV Resort. I’d be happy to set you up with our Sales Manager, Jamie Smith, for a personal resort tour so you can get a feel for what makes the Springs so special. You’re welcome to reply with a day and time that works best for you, or you can book directly using our online calendar (BOOK HERE) or call 778-871-3160. As you may know, our resort offers daily rentals, seasonal rentals, and RV lot ownership. If you’re exploring ownership, our brand-new Phase 3 lots range from $269,000 to over $300,000. We also have a selection of resale lots available; you can browse listings with photos and pricing (see details here). If you have any questions or would like to chat before your visit, I’m always here to help! Warmly, Kelsey Sales Assistant sales@springsrv.com Please note: Our Resort is recreational use only and does not allow full-time living.",
    "waitlist": "Hello {first_name}, Thank you for joining the waitlist at Springs RV Resort. We have received your details and will be in touch soon with more information Best regards, Kelsey Springs RV Resort sales@springsrv.com Please note: Our Resort is recreational use only and does not allow full-time living.",
    "pricelist": "Hi {first_name}, Thank you for your interest in the Springs RV Resort. My name is Kelsey, and I’m the Sales Assistant here. I’ve included our Phase 3 Price List & Lot Map so you can explore availability, financing options, and layout details. Our Phase 3 lots range from $269,000 to over $300,000. We also have lots for sale through our realtor—see prices and images via the provided link. The monthly maintenance fee is $210. I’d be happy to set you up with our Sales Manager, Jamie Smith, for a personal tour so you can experience what makes the Springs special. Please reply with a preferred time or book directly using our online calendar (BOOK HERE) or call 778-871-3160. If you have any questions or would like to chat before your visit, I’m here to help! Warmly, Kelsey Sales Assistant sales@springsrv.com Please note: Our Resort is recreational use only and does not allow full-time living.",
    "inquiries": "Hi {first_name}, Thank you for your interest in the Springs RV Resort. My name is Kelsey, and I’m the Sales Assistant here. I’ve included our Phase 3 Price List & Lot Map so you can explore availability, financing options, and layout details. Our Phase 3 lots range from $269,000 to over $300,000. We also have lots for sale through our realtor—see prices and images via the provided link. The monthly maintenance fee is $210. I’d be happy to set you up with our Sales Manager, Jamie Smith, for a personal tour so you can experience what makes the Springs special. Please reply with a preferred time or book directly using our online calendar (BOOK HERE) or call 778-871-3160. If you have any questions or would like to chat before your visit, I’m here to help! Warmly, Kelsey Sales Assistant sales@springsrv.com Please note: Our Resort is recreational use only and does not allow full-time living."
}

def rework_emails_to_prompt_completion(input_path: str, output_path: str) -> None:
    """
    Reads an emails JSON file (a list of records) and converts each record into a new JSONL file.
    Each output record contains:
        - "prompt": a string that combines the subject and body.
        - "completion": a baseline response selected based on the record's form (contact, waitlist, pricelist, inquiries).
    
    Parameters:
        input_path (str): Path to the input emails JSON file.
        output_path (str): Path to write the output JSONL file.
    """
    with open(input_path, "r") as infile:
        records = json.load(infile)
    
    with open(output_path, "w") as outfile:
        count = 0
        for rec in records:
            subject = rec.get("subject", "").strip()
            body = rec.get("body", "").strip()
            form = rec.get("form", "").strip().lower()
            # Construct prompt combining subject and body
            prompt = f"Subject: {subject}\n\nBody: {body}"
            # Choose the baseline response based on form; use a default message if the form is not in the mapping.
            completion = BASELINE_RESPONSE.get(form, "Thank you for contacting Springs RV Resort. We will respond shortly.")
            
            new_entry = {"prompt": prompt, "completion": completion}
            outfile.write(json.dumps(new_entry) + "\n")
            count += 1
    print(f"Converted {count} email records to prompt/completion format at {output_path}")

def convert_documents_clean_to_jsonl(input_path: str, output_path: str) -> None:
    """
    Reads a text file containing JSON objects (with possible comment lines starting with '//')
    and writes them into a JSONL file (one JSON object per line).
    
    Parameters:
        input_path (str): Path to the input text file.
        output_path (str): Path to write the output JSONL file.
    """
    count = 0
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                outfile.write(json.dumps(obj) + "\n")
                count += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}\n  {e}")
    print(f"Converted {count} JSON objects to JSONL format at {output_path}")


def fix_unescaped_quotes(line: str) -> str:
    """
    Naively fixes unescaped inner quotes in the "completion" field.
    This function finds the value of "completion" and escapes any
    double quotes that are not already escaped.
    """
    key = '"completion":'
    idx = line.find(key)
    if idx == -1:
        return line
    
    # Find the first and last quote that enclose the completion value.
    start = line.find('"', idx + len(key))
    end = line.rfind('"')
    if start == -1 or end == -1 or end <= start:
        return line

    # Extract the value and escape quotes that are not already escaped.
    value = line[start+1:end]
    fixed_value = re.sub(r'(?<!\\)"', r'\"', value)
    fixed_line = line[:start+1] + fixed_value + line[end:]
    return fixed_line

def sample_and_print(df, n=12000, random_state=42):
    """
    Samples n rows from the DataFrame using the given random_state,
    prints the first few rows of the sample, and returns the sampled DataFrame.
    
    :param df: Input DataFrame.
    :param n: Number of rows to sample (default: 100).
    :param random_state: Seed for random number generator (default: 42).
    :return: Sampled DataFrame.
    """
    sampled_data = df.sample(n=n, random_state=random_state)
    sampled_data.head()
    return sampled_data

def clean_newlines(text):
    """
    Replace multiple newline characters with a single newline.
    """
    if isinstance(text, str):
        return re.sub(r'\n+', '\n', text)
    return text

def clean_dataframe(df, columns):
    """
    Applies newline cleaning to the specified columns of the DataFrame.
    
    :param df: Input DataFrame.
    :param columns: List of column names to clean.
    :return: DataFrame with cleaned text.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_newlines, remove_curly_braces)
    return df

def reformat_prompt(raw_prompt):
    """
    Extracts key information from the raw prompt and rebuilds a clean formatted prompt.
    
    Expected raw prompt structure is similar to:
      Subject: <subject>
      Body: From: <sender> <email>
      City: <city>
      Phone: <phone>
      When would you like to come for a tour?:
      <tour_info>
      What are you interested in?
      <interests>
      Anything you'd like to ask us?
      <optional questions>
      ... (then the rest is UTM info and contact signature)
      
    Returns a new prompt with a clear structure.
    """
    # Remove unwanted UTM and signature noise.
    cleaned = re.sub(r'\nutm_[^:]+:.*', '', raw_prompt, flags=re.DOTALL)
    cleaned = re.sub(r'\ngclid:.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n--\s*\n.*', '', cleaned, flags=re.DOTALL)
    
    # # Extract key fields with regex.
    # subject = re.search(r'Subject:\s*(.*?)\n', cleaned)
    # subject = subject.group(1).strip() if subject else "N/A"
    
    # sender = re.search(r'From:\s*(.*?)\n', cleaned)
    # sender = sender.group(1).strip() if sender else "N/A"
    
    # city = re.search(r'City:\s*(.*?)\n', cleaned)
    # city = city.group(1).strip() if city else "N/A"
    
    # phone = re.search(r'Phone:\s*(.*?)\n', cleaned)
    # phone = phone.group(1).strip() if phone else "N/A"
    
    # tour = re.search(r'When would you like to come for a tour\?:\s*(.*?)\n', cleaned, re.DOTALL)
    # tour = tour.group(1).strip() if tour else "N/A"
    
    form_type = "Springs RV Resort Contact Form"
    interests = re.search(r'What are you interested in\?\s*(.*?)\n', cleaned, re.DOTALL)
    interests = interests.group(1).strip() if interests else "N/A"
    
    questions = re.search(r'Anything you\'d like to ask us\?\s*(.*?)\n', cleaned, re.DOTALL)
    questions = questions.group(1).strip() if questions else ""
    
    # Build a clean, structured prompt template.
    formatted = (
        f"Form Type: {form_type}\n"
        f"Interests: {interests}\n"
        f"Questions: {questions}"
    )
    return formatted

import re

def filter_friendly_rows(df, friendly_words, columns=['prompt', 'completion']):
    """
    Filters the DataFrame to keep only rows where the specified columns contain
    any of the friendly words (case-insensitive).
    
    :param df: Input DataFrame.
    :param friendly_words: List of words/phrases to search for.
    :param columns: List of column names to search within.
    :return: Filtered DataFrame.
    """
     # Create a regex pattern without word boundaries to better handle multi-word phrases.
    pattern = re.compile('(?:' + '|'.join(re.escape(word.lower()) for word in friendly_words) + ')')
    
    def row_has_friendly(text):
        return bool(pattern.search(str(text).lower()))
    
    mask = df[columns[0]].apply(row_has_friendly)
    for col in columns[1:]:
        if col in df.columns:
            mask = mask | df[col].apply(row_has_friendly)
    
    return df[mask]

import pandas as pd

def filter_out_nan(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Filters out rows with NaN values from the specified columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for NaN values. 
                        If None, checks all columns.
    
    Returns:
        pd.DataFrame: A DataFrame with rows containing NaN values removed.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

import pandas as pd

def save_df_to_jsonl(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves a Pandas DataFrame to a JSONL file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path to the output JSONL file.
    """
    try:
        # Convert the DataFrame to JSONL format and save to the specified file
        df.to_json(output_path, orient='records', lines=True)
        print(f"DataFrame successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to JSONL: {e}")

import re

def remove_curly_braces(text):
    """
    Removes all curly braces `{}` and their contents from the given text.

    :param text: The input string.
    :return: The cleaned string with all `{}` and their contents removed.
    """
    # Use regex to match and remove everything within curly braces, including the braces
    cleaned_text = re.sub(r"\{\{.*?\}\}", "", text)
    return cleaned_text.strip()

def clean_jsonl_file(input_file, output_file):
    """
    Cleans a JSONL file by removing all curly braces `{}` and their contents from the 'prompt' and 'completion' fields.

    :param input_file: Path to the input JSONL file.
    :param output_file: Path to save the cleaned JSONL file.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            # Clean 'prompt' and 'completion' fields
            data["prompt"] = remove_curly_braces(data["prompt"])
            data["completion"] = remove_curly_braces(data["completion"])
            # Write the cleaned data back to the output file
            outfile.write(json.dumps(data) + "\n")

if __name__ == "__main__":

    # Test cleaning function
    sample_text = "Hello, World! This is a sample text. Contact us at example@test.com. Call us at 604-557-6168."
    cleaning_conf = {
        "lower_case": True,
        "remove_punctuation": False  # Set False so demo phone number isn't stripped before formatting
    }
    print("Clean Text for BERT:", clean_text_for_bert(sample_text, cleaning_conf))
    
    # Test conversion of Instagram posts
    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/cache/insta/insta.txt"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/insta.jsonl"
    convert_insta_txt_to_jsonl(input_path, output_path)

    input_file = "/path/to/your/input.json"
    output_file = "/path/to/your/output.json"
    remove_empty_body_entries(input_file, output_file)
    print(f"Filtered output written to {output_file}")

    # Example usage: add "waitlist" as the form type to all entries
    input_file = "/path/to/your/input.json"
    output_file = "/path/to/your/output.json"
    form_type = "waitlist"  # Change as needed, e.g. "contact", "phase3_sales", etc.
    add_form_field(input_file, output_file, form_type)
    print(f'Updated entries with form="{form_type}" written to {output_file}')

    input_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website.jsonl"
    output_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website_company_voice.jsonl"
    voice = "Friendly, professional, and warm - reflecting Springs RV Resort's brand identity."
    add_company_voice(input_file, output_file, voice)
    print(f"Updated file with company_voice metadata written to {output_file}")

    emails_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/pre-output/emails_contact.jsonl"
    forms_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/forms.yaml"  # Your YAML file with baseline responses
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/output/emails_contact_enriched.jsonl"
    process_emails(emails_jsonl, forms_yaml, output_jsonl)
    print("Enriched email data written to", output_jsonl)

    instagram_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/instagram.jsonl"
    socials_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/socials_response.yaml"
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/enriched/instagram_enriched.jsonl"
    process_instagram_posts(instagram_jsonl, socials_yaml, output_jsonl)
    print("Enriched Instagram data written to", output_jsonl)

    website_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website.jsonl"
    socials_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/socials_response.yaml"
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/enriched/website_enriched.jsonl"
    process_website_posts(website_jsonl, socials_yaml, output_jsonl)
    print("Enriched website data written to", output_jsonl)

    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook_clean.txt"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook.json"
    with open(input_path, "r") as infile:
        text = infile.read()
    parsed_entries = parse_brandbook(text)
    with open(output_path, "w") as outfile:
        json.dump(parsed_entries, outfile, indent=4)
    print("Parsed brand book saved to", output_path)

    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/pre-output/gpt.json"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/training_data.jsonl"
    with open(input_path, "r") as infile:
        conversations = json.load(infile)
    with open(output_path, "w") as outfile:
        for conversation in conversations:
            example = convert_conversation_to_example(conversation)
            outfile.write(json.dumps(example) + "\n")

    # Load all json file in a dir
    data_dir = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    combined_data = load_all_jsonl_files(data_dir)
    print("Combined dataset count:", len(combined_data))

    # Load in and normalize jsonl
    file_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook.jsonl"
    normalized_brandbook = load_normalized_jsonl(file_path)
    print("Normalized DataFrame shape:", normalized_brandbook.shape)

    txt_directory = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    df_txt = load_txt_files_to_df(txt_directory)
    print("Loaded", len(df_txt), "text files")
    print(df_txt.head())

    # Split data function example
    data_dir = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    combined_data = load_all_jsonl_files(data_dir)  # or your combined DataFrame
    combined_df = pd.DataFrame(combined_data).fillna("")  # Ensure NaNs become empty strings
    train_df, test_df = split_train_test(combined_df,
                                         test_size=0.2,
                                         random_state=42)
    print("Train DataFrame shape:", train_df.shape)
    print("Test DataFrame shape:", test_df.shape)

    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook.jsonl"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook_converted.jsonl"
    convert_brandbook_file(input_path, output_path, default_prompt="Brandbook entry:")
    
    # convert email to json with prompt and completion
    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/pre-output/emails_contact.json"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/emails_contact_converted.jsonl"
    rework_emails_to_prompt_completion(input_path, output_path)

    # Convert clean docs to jsonl
    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/documents_clean.txt"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/documents_clean_converted.jsonl"
    convert_documents_clean_to_jsonl(input_path, output_path)

    # # Create a small data batch, Example usage:
    # sampled_data = sample_and_print(data_small)

    # # Clean \n\n newlines
    # # Example usage with your sampled_data DataFrame:
    # sampled_data = data_small.sample(n=100, random_state=42)  # Get 100 random rows
    # # Clean up the 'prompt' and 'completion' columns
    # sampled_data = clean_dataframe(sampled_data, ['prompt', 'completion'])

    # Example: reading a JSONL file, reformating each prompt, and saving the new JSONL:
    input_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/google_colab/emails_contact_converted.jsonl"
    output_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/google_colab/emails_contact_converted_clean.jsonl"

    # with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    #     for line in infile:
    #         record = json.loads(line)
    #         if 'prompt' in record:
    #             record['prompt'] = reformat_prompt(record['prompt'])
    #         outfile.write(json.dumps(record) + "\n")

    # print("Reformatted JSONL saved to:", output_file)

    # Friendly Words Example usage:
    friendly_words = [
        "Thank You", "understandable", "Definitely", "Absolutely", "Certainly",
        "Exactly", "Completely", "Quickly", "Fantastic", "Great", "Marvellous",
        "Excellent", "Enjoy", "Splendid", "Essential", "Generous", "Recommend",
        "Friendly", "Impressive", "Interesting", "Brilliant", "Exciting", "Terrific",
        "Fascinating", "Expert", "Favourite", "Ideal"
    ]

    # filter friendly rows
    filtered_data = filter_friendly_rows(data, friendly_words, columns=['prompt','completion'])
    print(filtered_data.head())

    # Filter out NaN values Example usage:
    # Assuming `df` is your DataFrame
    filtered_df = filter_out_nan(df, columns=['prompt', 'completion'])
    print(filtered_df.head())

    # convert dataframe to jsonl Example usage:
    # Assuming `df` is your DataFrame
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/cleaned_data.jsonl"
    save_df_to_jsonl(df, output_path)

    # remove curly braces from text Example usage
    example_text = "This is a test {{Order Number}} with some {{Placeholder}} text."
    cleaned_text = remove_curly_braces(example_text)
    print(cleaned_text)  # Output: "This is a test with some text."

    #remove curly braces form json Example usage
    input_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/attitude_data.jsonl"
    output_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/cleaned_attitude_data.jsonl"
    clean_jsonl_file(input_file, output_file)