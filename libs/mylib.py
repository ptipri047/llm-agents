import requests


class BaseClass:
    def __init__(self, myparam):
        print("base class")
        self.attr = myparam

    def mymethod(self, value):
        print(f"value is:{value}")
        print(f"attr: {self.attr}")


class SubClass(BaseClass):
    def __init__(self, myparam):
        super().__init__(myparam)
        print("i am in subclass")

    def submethod(self):
        print(f"in submethod: {self.attr}")

    def mycall(self, url):
        resp = requests.get(url)

        status = resp.status_code
        print(f"my status is: {status}")

        content = resp.content
        # print(f'content is {content}')
