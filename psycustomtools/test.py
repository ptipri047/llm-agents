import requests
url ='https://duckduckgo.com/'
resp = requests.get(url)
print(resp.text)