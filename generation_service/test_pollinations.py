import requests
import json
prompt = "Ambient cinematic soundtrack"
url = f"https://text.pollinations.ai/{prompt}"
print(requests.get(url).text)
