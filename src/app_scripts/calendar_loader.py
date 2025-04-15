import os
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Google Calendar API credentials
SCOPES = [os.getenv("SCOPES")]
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

# Authenticate with the Google Calendar API
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('calendar', 'v3', credentials=credentials)

def add_calendar_event(summary, description, start_time, end_time, time_zone="America/Vancouver"):
    """
    Add an event to the Google Calendar.

    Args:
        summary (str): Title of the event.
        description (str): Description of the event.
        start_time (datetime): Start time of the event.
        end_time (datetime): End time of the event.
        time_zone (str): Time zone of the event (default: "America/Vancouver").
    """
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': time_zone,
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': time_zone,
        },
    }

    # Insert the event into the primary calendar
    event_result = service.events().insert(calendarId='primary', body=event).execute()
    print(f"Event created: {event_result.get('htmlLink')}")

# Example usage
if __name__ == "__main__":
    # Define event details
    event_summary = "Follow-up with Elizabeth"
    event_description = "Discuss tour details and confirm availability."
    start_time = datetime.now() + timedelta(days=30)  # Event starts in 30 days
    end_time = start_time + timedelta(hours=1)  # Event lasts 1 hour

    # Add the event to the calendar
    add_calendar_event(event_summary, event_description, start_time, end_time)