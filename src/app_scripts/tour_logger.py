import json
import pandas as pd
import os
import csv
import datetime

TOUR_LOG_JSON = "/Users/blairjdaniel/Springs-AI/tour_logs/tour_log.json"
TOUR_LOG_CSV  = "/Users/blairjdaniel/Springs-AI/tour_logs/tour_log.csv"
TOUR_LOG_EXCEL = "/Users/blairjdaniel/Springs-AI/tour_logs/tour_log.xlsx"

def log_tour_details(subscription, email, date):
    # Create a base log entry from the event data.
    log_entry = {"subscription": subscription, "email": email, "date": date}
    
    # Unpack the tour details (date and time) from the subscription, if available.
    try:
        start_time = subscription.get("start_time")
        if start_time:
            if start_time.endswith("Z"):
                start_time = start_time[:-1]
            tour_dt = datetime.datetime.fromisoformat(start_time)
            log_entry["tour_date"] = tour_dt.date().isoformat()
            log_entry["tour_time"] = tour_dt.time().strftime("%H:%M")
        else:
            log_entry["tour_date"] = ""
            log_entry["tour_time"] = ""
    except Exception as e:
        log_entry["tour_date"] = ""
        log_entry["tour_time"] = ""
        print("Error parsing tour start_time:", e)
    
    # Log to JSON - append a new entry.
    try:
        logs = []
        if os.path.exists(TOUR_LOG_JSON):
            with open(TOUR_LOG_JSON, "r") as f:
                content = f.read().strip()
            if content:
                logs = json.loads(content)
        # Always append instead of updating last entry.
        logs.append(log_entry)
        with open(TOUR_LOG_JSON, "w") as f:
            json.dump(logs, f, indent=4)
        print("Logged tour details to JSON successfully.")
    except Exception as e:
        print("Error logging tour details to JSON:", e)
    
    # Extract booking user from event_memberships.
    if isinstance(subscription, dict):
        memberships = subscription.get("event_memberships", [])
        if memberships and len(memberships) > 0:
            user_val = memberships[0].get("user_name", "No User")
        else:
            user_val = "No User"
    else:
        user_val = "No User"
    
    # Log to CSV - append a new row.
    try:
        rows = []
        if os.path.exists(TOUR_LOG_CSV):
            with open(TOUR_LOG_CSV, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        row_data = {
            "name": subscription.get("name", "No Name") if isinstance(subscription, dict) else "No Name",
            "user": user_val,
            "email": email,
            "date": date,
            "tour_date": log_entry.get("tour_date", ""),
            "tour_time": log_entry.get("tour_time", "")
        }
        # Append new row.
        rows.append(row_data)
        with open(TOUR_LOG_CSV, "w", newline="") as f:
            fieldnames = ["name", "user", "email", "date", "tour_date", "tour_time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("Logged tour details to CSV successfully.")
    except Exception as e:
        print("Error logging tour details to CSV:", e)
    
    # Log to Excel - append a new row.
    try:
        try:
            df = pd.read_excel(TOUR_LOG_EXCEL)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["name", "user", "email", "date", "tour_date", "tour_time"])
        new_data = {
            "name": subscription.get("name", "No Name") if isinstance(subscription, dict) else "No Name",
            "user": user_val,
            "email": email,
            "date": date,
            "tour_date": log_entry.get("tour_date", ""),
            "tour_time": log_entry.get("tour_time", "")
        }
        # Append new row to DataFrame.
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_excel(TOUR_LOG_EXCEL, index=False)
        print("Logged tour details to Excel successfully.")
    except Exception as e:
        print("Error logging tour details to Excel:", e)