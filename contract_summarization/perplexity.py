import requests

# Set your API key here
API_KEY = "--------"

url = "https://api.perplexity.ai/chat/completions"
payload = {
    "model": "sonar",
    "messages": [
        {
            "role": "system",
            "content": "Be precise and concise."
        },
        {
            "role": "user",
            "content": "What news is there about cats in MexicO??"
        }
    ],
    "max_tokens": 2000,

}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.text)
print(response.content)
