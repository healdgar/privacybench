import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEY is not set!")
else:
    print("API Key loaded:", openai_api_key[:10] + "...")

client = OpenAI(api_key=openai_api_key)
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        response_format={"type": "text"},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response.choices[0].message.content.strip())
except Exception as e:
    print("Error during OpenAI call:", e)
