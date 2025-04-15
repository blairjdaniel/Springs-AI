def generate_few_shot_prompt(email_text, sender_name, email_category):
    """
    Generate a few-shot prompt to guide the model's response style.
    """

    # Few-shot examples
    few_shot_examples = {
        "contact": """
Hi {sender_name},

Thanks so much for reaching out! My name is Kelsey, and I’m the Sales Assistant here at Springs RV Resort.

I’d be happy to set you up with our Sales Manager, Jamie Smith, for a personal resort tour so you can get a feel for what makes the Springs so special. You’re welcome to reply with a day and time that works best for you, or you can book directly using our online calendar: [BOOK HERE](https://calendly.com/springs-rv-resort/springs-rv-resort-sales-meeting?month=2025-04/) or call 778-871-3160.

If you have any questions or would like to chat before your visit, I’m always here to help!

Warmly,  
Kelsey  
Sales Assistant  
sales@springsrv.com  

Please note: Our Resort is recreational use only and does not allow full-time living.
""",
        "waitlist": """
Hello {sender_name},

Thank you for joining the waitlist at Springs RV Resort.

We have received your details and will be in touch soon with more information.

Best regards,  
Kelsey  
Springs RV Resort  
sales@springsrv.com  

Please note: Our Resort is recreational use only and does not allow full-time living.
""",
        "pricelist": """
Hi {sender_name},

Thank you for your interest in the Springs RV Resort. My name is Kelsey, and I’m the Sales Assistant here at Springs RV Resort.

I’ve included our Phase 3 Price List & Lot Map ([click here](https://springsrv.com/phase-3-lots-for-sale)) so you can explore availability, financing options, and layout details at your own pace. Our Phase 3 lots range from $269,000 to over $300,000. We also have lots that are for sale through our realtor, and you can see prices and images [here](https://springsrv.com/phase-3-lots-for-sale/). The monthly maintenance fee is $210, which includes your property taxes and keeps the resort and pools looking beautiful!  

I’d be happy to set you up with our Sales Manager, Jamie Smith, for a personal resort tour so you can get a feel for what makes the Springs so special. You’re welcome to reply with a day and time that works best for you, or you can use our online calendar to book directly: [BOOK HERE](https://calendly.com/springs-rv-resort/springs-rv-resort-sales-meeting?month=2025-04/) or call 778-871-3160 to find a time to come out.  

If you have any questions or would like to chat before your visit, I’m always here to help!

Warmly,  
Kelsey  
Sales Assistant  
Springs RV Resort  
sales@springsrv.com
"""
    }

    # Get the relevant examples for the email category
    examples = few_shot_examples.get(email_category, few_shot_examples["contact"])  # Default to "contact"

    # Replace placeholders with actual values
    prompt = examples.format(sender_name=sender_name, email_text=email_text)
    return prompt