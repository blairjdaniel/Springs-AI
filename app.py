import sys
sys.path.append('./src')
sys.path.append('./utils')
import os
import re
import yaml
import requests
from flask import Flask, render_template, request
from imap_tools import MailBox, AND
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Initialize the tokenizer and model
gpt_tokenizer = AutoTokenizer.from_pretrained("/Users/blairjdaniel/AI-Assistant-Springs/models/gpt")  # Replace "" with your model name if different
gpt_model = AutoModelForCausalLM.from_pretrained("/Users/blairjdaniel/AI-Assistant-Springs/models/gpt")  # Replace "" with your model name if different
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from my_prompt_engineering.few_shot import generate_few_shot_prompt
from app_scripts.calendar_loader import add_calendar_event
from my_prompt_engineering.few_shots_email_responder import process_email_and_generate_response
from utils.config_loader import load_baseline_responses
from utils.email_classifier import classify_email, load_email_classifier
from utils.follow_up import check_for_follow_ups
from utils.helpers import extract_sender_name, send_email, process_email
from utils.model_loader import load_gpt_model
from utils.generate_response import generate_response


# Load environment variables
load_dotenv()

# Email and IMAP configuration
IMAP_SERVER = os.getenv("IMAP_SERVER")
SPRINGS_EMAIL_USERNAME = os.getenv("SPRINGS_EMAIL_USERNAME")
SPRINGS_EMAIL_PASSWORD = os.getenv("SPRINGS_EMAIL_PASSWORD")

# Flask app setup
app = Flask(__name__)

# Load baseline responses from YAML file
with open("/Users/blairjdaniel/AI-Assistant-Springs/config/baseline_responses.yaml", "r") as f:
    baseline_responses = yaml.safe_load(f)

print("Loaded baseline_responses:", baseline_responses)

# Label mapping for email classification
label_mapping = {
    "LABEL_0": "contact",
    "LABEL_1": "waitlist",
    "LABEL_2": "pricelist",
    "LABEL_3": "inquiries"
}

# Validate baseline responses
for category in label_mapping.values():
    if category not in baseline_responses["forms"]:
        print(f"Warning: Missing baseline response for category '{category}'")

@app.route("/fetch_calendly_tours", methods=["POST"])
def fetch_calendly_tours():
    """
    Trigger the Calendly app to fetch and log tours.
    """
    calendly_app_url = "http://127.0.0.1:5001/log_calendly_tours"  # URL of the Calendly app
    try:
        response = requests.post(calendly_app_url)
        if response.status_code == 200:
            result = response.json()
            return render_template("check_emails.html", result=result["message"])
        else:
            return render_template("check_emails.html", result="Failed to fetch tours from Calendly.")
    except requests.exceptions.RequestException as e:
        return render_template("check_emails.html", result=f"Error: {e}")


@app.route("/check_emails", methods=["GET", "POST"])
def check_emails():
    result = None
    processed_emails = []
    ignored_senders = ["success@docusign.com", "dse@camail.docusign.net", "jodi.dave@yahoo.ca"]
    target_subject = [
        "Springs RV Resort Contact Us",
        "Springs RV Resort Join the waitlist",
        "RV Lot Phase 3 Price List - Springs RV Resort"
    ]
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

    if request.method == "POST":
        # Specify the model path for DistilBERT
        distilbert_model_path = "/Users/blairjdaniel/AI-Assistant-Springs/models/distilbert-email-classifier"

        # Load the email classifier
        classifier = load_email_classifier(distilbert_model_path)
        if not (IMAP_SERVER and SPRINGS_EMAIL_USERNAME and SPRINGS_EMAIL_PASSWORD):
            result = "IMAP server credentials not set."
        else:
            with MailBox(IMAP_SERVER).login(SPRINGS_EMAIL_USERNAME, SPRINGS_EMAIL_PASSWORD, initial_folder="INBOX") as mailbox:
                print("Connected to the IMAP server successfully.")
                criteria = AND(all=True) if TEST_MODE else AND(seen=False)
                emails = list(mailbox.fetch(criteria, limit=20))
                print(f"Fetched {len(emails)} emails.")

                for msg in emails:
                    sender = msg.from_
                    email_text = msg.text
                    subject = msg.subject

                    print(f"Processing email from {sender} with subject: {subject}")

                    # Skip ignored senders
                    if sender.lower() in ignored_senders:
                        print(f"Skipping email from ignored sender: {sender}")
                        continue

                    # Skip emails with subjects not in target_subject
                    if not any(target in subject for target in target_subject):
                        print(f"Skipping email with subject: {subject}")
                        continue

                    # Process the email (classification, extraction, etc.)
                    email_category = classify_email(subject, classifier, label_mapping)
                    print(f"Classified email as category: {email_category}")

                    tailored_response = generate_response(email_text, sender, email_category, gpt_tokenizer, gpt_model)

                    result = process_email_and_generate_response(email_text, sender, gpt_tokenizer, gpt_model)

                    #Extract the responses and generated response
                    extracted_response = result["responses"]
                    generated_response = result["generated_response"]

                    processed_emails.append({
                        "sender": sender,
                        "subject": subject,
                        "category": email_category,
                        "response": tailored_response,
                        "tour_response": extracted_response.get("tour_response"),
                        "ask_response": extracted_response.get("ask_response")
                    })

                    print(f"Processed email with generated response:\n{generated_response}")
                    

                    if not TEST_MODE:
                        mailbox.flag(msg.uid, '//Seen', True)

                result = f"Processed {len(processed_emails)} emails."

    return render_template("check_emails.html", result=result, processed_emails=processed_emails, baseline_responses=baseline_responses)



if __name__ == "__main__":
    app.run(debug=True)