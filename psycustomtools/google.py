#https://scrapfly.io/blog/how-to-scrape-google/
from collections import defaultdict
from urllib.parse import quote
from httpx import Client
from parsel import Selector
import json

class Google:
    def __init__(self):
        super().__init__()
        print('enter google')
        
        # 1. Create HTTP client with headers that look like a real web browser
        self.client = Client(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6",
            },
            follow_redirects=True,
            http2=True,  # use HTTP/2
        )

    '''
        ****** Normal query
    '''
    def parse_search_results(self, selector: Selector):
        """parse search results from google search page"""
        results = []
        for box in selector.xpath(
            "//h1[contains(text(),'Search Results')]/following-sibling::div[1]/div"
        ):
            title = box.xpath(".//h3/text()").get()
            url = box.xpath(".//h3/../@href").get()
            text = "".join(box.xpath(".//div[@data-sncf]//text()").getall())
            if not title or not url:
                continue
            url = url.split("://")[1].replace("www.", "")
            results.append(title, url, text)
        return results

    def scrape_search(self, query: str, page=1):
        """scrape search results for a given keyword"""
        # retrieve the SERP
        url = f"https://www.google.com/search?hl=en&q={quote(query)}" + (
            f"&start={10*(page-1)}" if page > 1 else ""
        )
        print(f"scraping {query=} {page=}")
        results = defaultdict(list)
        response = self.client.get(url)
        assert response.status_code == 200, f"failed status_code={response.status_code}"
        # parse SERP for search result data
        selector = Selector(response.text)
        results["search"].extend(self.parse_search_results(selector))
        return dict(results)

    '''
        # How to Scrape Google SEO Rankings
    check_ranking(
    keyword="scraping instagram", 
    url_match="scrapfly.com/blog/",
)
    '''
    def check_ranking(self,keyword: str, url_match: str, max_pages=3):
        """check ranking of a given url (partial) for a given keyword"""
        rank = 1
        for page in range(1, max_pages + 1):
            results = self.scrape_search(keyword, page=page)
            for (title, result_url, text) in results["search"]:
                if url_match in result_url:
                    print(f"rank found:\n  {title}\n  {text}\n  {result_url}")
                    return rank
                rank += 1
        return None


    '''
    How to Scrape Google Keyword Data
    A big part of SEO is keyword research - understanding what people are searching for and how to optimize content based on these queries.

    When it comes to Google search scraping, the "People Also Ask" and "Related Searches" sections can be used in keyword research:
    '''
    def parse_related_search(self,selector: Selector):
        """get related search keywords of current SERP"""
        results = []
        for suggestion in selector.xpath(
            "//div[div/div/span[contains(text(), 'Related searches')]]/following-sibling::div//a"
        ):
            results.append("".join(suggestion.xpath(".//text()").getall()))
        return results


    def parse_people_also_ask(self, selector: Selector):
        """get people also ask questions of current SERP"""
        return selector.css(".related-question-pair span::text").getall()

    def scrape_search_peopleask(self, query: str, page=1):
        """scrape search results for a given keyword"""
        # retrieve the SERP
        url = f"https://www.google.com/search?hl=en&q={quote(query)}" + (f"&start={10*(page-1)}" if page > 1 else "")
        print(f"scraping {query=} {page=}")
        results = defaultdict(list)
        response = self.client.get(url)
        assert response.status_code == 200, f"failed status_code={response.status_code}"
        # parse SERP for search result data
        selector = Selector(response.text)
        results["related_search"].extend(self.parse_related_search(selector))
        results["people_also_ask"].extend(self.parse_people_also_ask(selector))
        return dict(results)




# example use: scrape 3 pages: 1,2,3
google = Google()
for page1 in [1, 2, 3]:
    results1 = google.scrape_search("scrapfly blog", page=page1)
    for result1 in results1["search"]:
        print(result1)

# Example use: 
results = google.scrape_search_peopleask("scraping instagram")
print(json.dumps(results, indent=2))