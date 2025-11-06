import requests
import certifi

API_KEY = "AIzaSyDIDWnCjTCsK790Bktdqo23Xnc_1bU-EsY"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

prompt_text = "Use the following context to answer the user in a friendly tone: ... \nUser: I feel stressed about exams."

data = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": prompt_text}]  # ✅ Must be a dict with key "text"
        }
    ]
}

headers = {
    "x-goog-api-key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(ENDPOINT, json=data, headers=headers, verify=certifi.where())
print(response.status_code)
print(response.text)
