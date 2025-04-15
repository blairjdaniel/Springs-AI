# helpers.py
import re
from apscheduler.schedulers.background import BackgroundScheduler

def extract_sender_name(email_body):
    """
    Extract the first name from the email body.
    Handles both 'From:' and 'Name:' formats.
    """
    from_match = re.search(r"From:\s*([\w'-]+)", email_body, re.IGNORECASE)
    if from_match:
        return from_match.group(1).strip()

    name_match = re.search(r"Name:\s*([\w'-]+)", email_body, re.IGNORECASE)
    if name_match:
        return name_match.group(1).strip()

    return "there"

def send_email(recipient, email_body):
    """
    Simulate sending an email.
    """
    print(f"Sending email to {recipient} with body:\n{email_body}")



# Process email content
def process_email(email_text):
    if "tour" in email_text.lower():
        return "Hi, I’ve confirmed your tour. Let me know if you have any questions!"
    elif "seasonal rental" in email_text.lower():
        return "Hi, I’d be happy to provide more details about our seasonal rentals. Let me know what you’re looking for!"
    elif "Phase 3" in email_text.lower():
        return "Hi, I’d be happy to provide more information about Phase 3. Let me know if you’d like to schedule a tour!"
    else:
        return "Hi, let me know how I can assist you further!"
    
def send_email(recipient, email_body):
    print(f"Sending email to {recipient} with body:\n{email_body}")

# Define the check_for_follow_ups function
def check_for_follow_ups():
    """
    Function to check for follow-ups.
    This is a placeholder implementation.
    """
    print("Checking for follow-ups...")

# Schedule follow-up checks
scheduler = BackgroundScheduler()
scheduler.add_job(check_for_follow_ups, 'interval', days=1)
scheduler.start()    

