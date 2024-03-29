from dotenv import load_dotenv

load_dotenv()

from libs.waitproxyplugin import SleepInRequests

import proxy
import requests
import time

'''
from proxy.common import utils

proxy_port = utils.get_available_port()
with proxy.Proxy(
        [
            "--sleeptime",
            '2'
        ],
        plugins=[SleepInRequests],
    ):
    print("here")
    resp = requests.get(url="http://google.fr")
    print("there")

'''
if __name__ == "__main__":
    from proxy.common import utils

    proxy_port = utils.get_available_port()
    with proxy.Proxy(
        [
            "--sleeptime",
            '2'
        ],
        plugins=[SleepInRequests],
    ):
        print("here")
        resp = requests.get(url="http://google.fr")
        print("there")
