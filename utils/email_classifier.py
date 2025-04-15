# email_classifier.py
from transformers import pipeline, AutoConfig

def load_email_classifier(model_path):
    """
    Load the email classification model.
    """
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return pipeline("text-classification", model=model_path, tokenizer=model_path, config=config)

def classify_email(subject, classifier, label_mapping, confidence_threshold=0.6):
    """
    Classify the email subject into a category.
    """
    result = classifier(subject)
    print(f"Raw classifier result for subject '{subject}': {result}")

    if not result or "label" not in result[0]:
        print(f"Classifier returned an empty or invalid result for subject: '{subject}'")
        return "contact"

    predicted_label = result[0]["label"]
    confidence_score = result[0]["score"]

    if confidence_score < confidence_threshold:
        print(f"Low confidence ({confidence_score}) for subject: '{subject}'. Defaulting to 'contact'.")
        return "contact"

    return label_mapping.get(predicted_label, "contact")