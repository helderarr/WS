import requests

search_url = "https://api.bing.microsoft.com/v7.0/search"
subscription_key = ""
search_term = "nouvelle usine"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()

print(search_results)