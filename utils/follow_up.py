# Check for follow-ups
def check_for_follow_ups():
    def get_emails_without_response(days):
        return [
            {"recipient": "example@example.com", "subject": "Inquiry about pricing"},
            {"recipient": "test@test.com", "subject": "Follow-up on waitlist"},
            {"recipient": "user@user.com", "subject": "Follow-up on tour dates"}
        ]

    emails_to_follow_up = get_emails_without_response(days=3)
    for email in emails_to_follow_up:
                follow_up_email = generate_follow_up_email(email)
        
    # Define the generate_follow_up_email function
    def generate_follow_up_email(email):
        subject = email.get("subject", "Follow-up")
        recipient = email.get("recipient", "there")
        return f"Hi {recipient},\n\nThis is a follow-up regarding your previous inquiry: '{subject}'. Please let us know if you have any further questions or need assistance.\n\nBest regards,\nSprings RV Resort Team"
    #send_email(email['recipient'], follow_up_email)    