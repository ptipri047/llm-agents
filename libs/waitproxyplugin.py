# -*- coding: utf-8 -*-
"""
proxy.py
~~~~~~~~
"""

from proxy.http.proxy import HttpProxyBasePlugin
from proxy.common.flag import flags
from proxy.http.parser import HttpParser
from proxy.common.utils import text_
import re
import json
import logging
from typing import Any, Dict, List, Optional

"""from http import httpStatusCodes
from http.proxy import HttpProxyBasePlugin
from common.flag import flags
from http.parser import HttpParser
from common.utils import text_
from http.exception import HttpRequestRejected"""

import time

logger = logging.getLogger(__name__)

# See adblock.json file in repository for sample example config
flags.add_argument(
    "--sleeptime",
    type=int,
    default="",
    help="amount of sleep between requests",
)


class SleepInRequests(HttpProxyBasePlugin):
    """do some sleep."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sleeptime = self.flags.sleeptime

    def handle_upstream_chunk(self, chunk: memoryview) -> memoryview:
        if "togeth" in self.url:
            time.sleep(2.0)
            print("proxy going to sleep for 2s")
        return chunk    

    def handle_client_request(
        self,
        request: HttpParser,
    ) -> Optional[HttpParser]:
        # determine host
        request_host = None
        try:
            if request.host:
                request_host = request.host
            elif request.headers and b"host" in request.headers:
                request_host = request.header(b"host")

            if not request_host:
                logger.error("Cannot determine host")
                return request

            # build URL
            self.url = f"{request_host.decode("utf-8")}"
            print(f"url connection to proxy: {self.url} ")

        except Exception as e:
            print(f"received exception in waitproxy: {e}")

        return request
