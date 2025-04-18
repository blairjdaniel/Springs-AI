import sys
sys.path.append('./src')

import requests
from urllib.parse import quote_plus
from app_scripts.tour_logger import log_tour_details
from app_scripts.calendar_loader import add_calendar_event

from dotenv import load_dotenv
import os
import datetime

# Load environment variables
load_dotenv()

# Use personal access token from .env rather than retrieving one via OAuth.
CALENDLY_API = os.getenv("CALENDLY_API")
CALENDLY_ORG_ID = os.getenv("CALENDLY_ORG_ID")
CALENDLY_WEBHOOK_ID = os.getenv("CALENDLY_WEBHOOK_ID")

def get_calendly_organization():
    if not CALENDLY_ORG_ID:
        print("Environment variable CALENDLY_ORG_ID not set.")
        return None
    organization_url = f"https://api.calendly.com/organizations/{CALENDLY_ORG_ID}"
    print("Organizational URL:", organization_url)
    return organization_url

def fetch_calendly_events():
    org_url = get_calendly_organization()
    if not org_url:
        print("Cannot fetch events without organization info.")
        return []
    encoded_org_url = quote_plus(org_url)
    url = f"https://api.calendly.com/scheduled_events?organization={encoded_org_url}"
    headers = {
        "Authorization": f"Bearer {CALENDLY_API}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        events = response.json().get("collection", [])
        print(f"Fetched {len(events)} events.")
        return events
    else:
        print(f"Error fetching events: {response.status_code}, {response.text}")
        return []

def create_webhook_subscription(callback_url, events, scope="organization"):
    """
    Create a webhook subscription using your personal access token.
    
    Parameters:
      callback_url (str): The URL where Calendly will send event notifications.
      events (list): A list of events to subscribe to, e.g., ["invitee.created", "invitee.canceled"].
      scope (str): The subscription scope ("organization" or "user").
      
    Returns:
      dict: The created webhook subscription data.
    """
    org_url = get_calendly_organization()
    if not org_url:
        print("Organization URI is not available.")
        return None

    payload = {
        "url": callback_url, 
        "events": events,
        "scope": scope,
        "organization": org_url
    }
    headers = {
        "Authorization": f"Bearer {CALENDLY_API}",
        "Content-Type": "application/json"
    }
    subscription_url = "https://api.calendly.com/webhook_subscriptions"
    response = requests.post(subscription_url, json=payload, headers=headers)
    if response.status_code == 201:
        print("Webhook subscription created successfully!")
        return response.json()
    elif response.status_code == 409:
        print(f"Webhook subscription already exists: {response.text}")
        # Optionally, return a value or fetch the existing subscription here.
        return {"already_exists": True, "details": response.json()}
    else:
        print(f"Error creating subscription: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    events = fetch_calendly_events()
    for event in events:
        print(event)
        log_tour_details(event, "sales@springsrv.com", datetime.datetime.now().isoformat())
    
    
    # Replace with your desired callback URL
    callback_url = "https://your_callback_url.com/webhook"
    subscribed_events = ["invitee.created", "invitee.canceled"]
    subscription = create_webhook_subscription(callback_url, subscribed_events, scope="organization")
    print(subscription)

# # Save subscription details using tour_logger once successful.
#     if subscription:
#         email = "sales@springsrv.com"  # Or obtain this from a configuration/module
#         date = datetime.datetime.now().isoformat()
#         log_tour_details(subscription, email, date)