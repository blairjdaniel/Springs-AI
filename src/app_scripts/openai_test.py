from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set the OpenAI API key
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
# Create a chat completion
completion = client.chat.completions.create(
    model="gpt-4o-mini",  # Use the correct model name
    messages=[
        {"role": "user", "content": "write a haiku about AI"}
    ]
)

# Print the response
print(completion.choices[0].message["content"])