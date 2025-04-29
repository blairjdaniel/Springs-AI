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
gpt_tokenizer = AutoTokenizer.from_pretrained("/Users/blairjdaniel/Springs-AI/models/gpt")  # Replace "" with your model name if different
gpt_model = AutoModelForCausalLM.from_pretrained("/Users/blairjdaniel/Springs-AI/models/gpt")  # Replace "" with your model name if different
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

# Import our Calendly functions and tour logger
from calendly import fetch_calendly_events
from src.app_scripts.tour_logger import log_tour_details
import openai

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
load_dotenv()

# Email and IMAP configuration
IMAP_SERVER = os.getenv("IMAP_SERVER")
SPRINGS_EMAIL_USERNAME = os.getenv("SPRINGS_EMAIL_USERNAME")
SPRINGS_EMAIL_PASSWORD = os.getenv("SPRINGS_EMAIL_PASSWORD")

# Flask app setup
app = Flask(__name__)

# Load baseline responses from YAML file
with open("/Users/blairjdaniel/Springs-AI/config/baseline_responses.yaml", "r") as f:
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
    Fetch tours from Calendly and log them using tour_logger.
    """
    try:
        events = fetch_calendly_events()
        for event in events:
            # Log each event using tour_logger.
            log_tour_details(event, "sales@springsrv.com", datetime.now().isoformat())
        result_message = f"Fetched and logged {len(events)} tour event(s)."
    except Exception as e:
        result_message = f"Error fetching or logging tours: {e}"
    return render_template("check_emails.html", result=result_message)


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
        distilbert_model_path = "/Users/blairjdaniel/Springs-AI/models/distilbert-email-classifier"

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

                    #result = process_email_and_generate_response(email_text, sender, gpt_tokenizer, gpt_model)

                    # #Extract the responses and generated response
                    # extracted_response = result["responses"]
                    # generated_response = result["generated_response"]

                    processed_emails.append({
                        "sender": sender,
                        "subject": subject,
                        "category": email_category,
                        "response": tailored_response,
                        "openai_response": None
                        # "tour_response": extracted_response.get("tour_response"),
                        # "ask_response": extracted_response.get("ask_response")
                    })

                    # print(f"Processed email with generated response:\n{generated_response}")
                    

                    if not TEST_MODE:
                        mailbox.flag(msg.uid, '//Seen', True)

                result = f"Processed {len(processed_emails)} emails."

    return render_template("check_emails.html", result=result, processed_emails=processed_emails, baseline_responses=baseline_responses)

@app.route("/openai_reply", methods=["POST"])
def openai_reply():
    """
    Generate a reply using OpenAI assistant for a specific email.
    """
    try:
        # Get email content and sender from the request
        email_text = request.form.get("email_text", "")
        sender = request.form.get("sender", "")
        if not email_text or not sender:
            return {"error": "Missing email content or sender"}, 400

        # Call OpenAI's API to generate a reply
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
             messages=[
                {"role": "system", "content": (
                    "You are a friendly, professional sales assistant at Springs RV Resort. " 
                    "Always include the Calendly scheduling link (https://calendly.com/springsrv) in your responses. "
                    "Format your reply similar to the following YAML baseline responses."
                )},
                {"role": "system", "content": (
                    "Example baseline response:\n"
                    "subject: Re: Contact\n"
                    "body: |\n"
                    "  Hi {{customer_name}},\n"
                    "  Thank you for reaching out. Please use the following link to schedule a tour: https://calendly.com/springsrv\n"
                    "  We look forward to welcoming you at Springs RV Resort.\n"
                    "  Best regards,\n"
                    "  Kelsey\n"
                )},
                {"role": "user", "content": f"Write a reply email to the following customer inquiry as the sales assistant: {email_text}"}
    ],
            max_tokens=250,
            temperature=0.7
)

        print(response)

        assistant_reply = response['choices'][0]['message']['content'].strip()

        return {"reply": assistant_reply}, 200
    except Exception as e:
        print(f"Error generating OpenAI reply: {e}")
        return {"error": str(e)}, 500
    
@app.route("/send_email", methods=["POST"])
def send_email_route():
    """
    Send an email reply to the original sender.
    """
    try:
        recipient = request.form.get("recipient", "")
        subject = request.form.get("subject", "")
        body = request.form.get("body", "")
        if not recipient or not subject or not body:
            return {"error": "Missing recipient, subject, or body"}, 400

        # Call the send_email helper function.
        send_email(recipient, subject, body)
        return {"status": "Email sent successfully"}, 200
    except Exception as e:
        print(f"Error sending email: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)