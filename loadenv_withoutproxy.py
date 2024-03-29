
# Get-ChildItem -Path Env:
from dotenv import load_dotenv
import os
#result = load_dotenv('./.env_withoutproxy', override= True)
#print(result)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

print(f'base-{os.getenv("http_proxy")}-')
