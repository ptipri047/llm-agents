'''
https://medium.com/@greyboi/ddgsearch-search-duckduckgo-scrape-the-results-in-python-18f5265f1aa6
https://www.geeksforgeeks.org/website-summarizer-using-bart/
'''
# pylint: disable=missing-function-docstring
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv('../.env_withoutproxy')
import requests
from bs4 import BeautifulSoup
import re

from duckduckgo_search import DDGS
import pprint
from scrapy.crawler import CrawlerProcess
import scrapy
import bs4



class MySpider(scrapy.Spider):
    '''
    This is the spider that will be used to crawl the webpages. 
    We give this to the scrapy crawler.
    '''
    name = 'myspider'
    start_urls = None
    clean_with_llm = False
    results = []

    def __init__(self, start_urls, clean_with_llm, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.clean_with_llm = clean_with_llm

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        type(response)
        if 200 == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            excludeList = ['disclaimer', 'cookie', 'privacy policy']
            includeList = soup.find_all(
                ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            elements = [element for element in includeList if not any(
                keyword in element.get_text().lower() for keyword in excludeList)]
            text = " ".join([element.get_text()
                            for element in elements])
            text = re.sub(r'\n\s*\n', '\n', text)
            return self.summarize(text)
        else:
            return "Error in response"

    def splitTextIntoChunks(self,text, chunk_size=1024):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks



    def summarize(self,text, chunk_size=1024, chunk_summary_size=128):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks = self.splitTextIntoChunks(text, chunk_size)
        
        summaries = []
        for chunk in chunks:
            size = chunk_summary_size
            if(len(chunk) < chunk_summary_size):
                size = len(chunk)/2
            summary = summarizer(chunk, min_length=1, max_length=size)[0]["summary_text"]
            summaries.append(summary)
        
        concatenated_summary = ""
        for summary in summaries:
            concatenated_summary += summary + " "

        return concatenated_summary    
    
        
class DuckDuckGo:
    def __init__(self):
        pass    

    def search(self, query):   
        results = DDGS().text(query, max_results=5)
        pprint.pp(results)

        urls = [res['href'] for res in results]
        process = CrawlerProcess()
        process.crawl(MySpider, urls, False)
        process.start()

        # here the spider has finished downloading the pages and cleaning them up
        return MySpider.results 
        
result = DuckDuckGo().search('PSG') 
pprint.pp(result)    