# Define file paths for logging
import json
import pandas as pd
import os
import csv
import re
import requests

TOUR_LOG_JSON = "/Users/blairjdaniel/AI-Assistant-Springs/tour_logs/tour_log.json"
TOUR_LOG_CSV = "/Users/blairjdaniel/AI-Assistant-Springs/tour_logs/tour_log.csv"
TOUR_LOG_EXCEL = "/Users/blairjdaniel/AI-Assistant-Springs/tour_logs/tour_log.xlsx"

def log_tour_details(name, email, date):
    # Log to JSON
    try:
        with open(TOUR_LOG_JSON, "r") as f:
            tour_log = json.load(f)
    except FileNotFoundError:
        tour_log = []

    tour_log.append({"name": name, "email": email, "date": date})
    with open(TOUR_LOG_JSON, "w") as f:
        json.dump(tour_log, f, indent=4)

    # Log to CSV
    file_exists = os.path.isfile(TOUR_LOG_CSV)
    with open(TOUR_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email", "date"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"name": name, "email": email, "date": date})

    # Log to Excel
    try:
        df = pd.read_excel(TOUR_LOG_EXCEL)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["name", "email", "date"])
    new_entry = pd.DataFrame([{"name": name, "email": email, "date": date}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(TOUR_LOG_EXCEL, index=False)

def extract_sender_email(email_body):
    match = re.search(r"From:.*<([\w\.-]+@[\w\.-]+)>", email_body)
    if match:
        return match.group(1).strip()
    return "unknown@example.com"

    from dateutil.parser import parse

def parse_tour_date(tour_response):
    try:
        return parse(tour_response)
    except ValueError:
        raise ValueError(f"Unable to parse tour date: {tour_response}")
    
