import mechanicalsoup
from bs4 import BeautifulSoup
import webbrowser

# Function to get content of multiple URLs without HTML tags
class Browse:
    def __init__(self):
        self.browser = mechanicalsoup.StatefulBrowser()
        self.browser.set_user_agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36')
    

    def debug_form_elements(self,url):
        print('here')    
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        
        # Get the current page
        current_page = browser.get_current_page()
        self.display_page_content(current_page)
        
        # Find all form elements
        forms = current_page.find_all('form')
        
        for form in forms:
            print("Form:")
            # Print form attributes
            print("  Method:", form.get('method'))
            print("  Action:", form.get('action'))
            
            # Find and print all input elements within the form
            input_elements = form.find_all('input')
            if input_elements:
                print("  Input elements:")
                for input_element in input_elements:
                    print("    Name:", input_element.get('name'))
                    print("    Type:", input_element.get('type'))
                    print("    Value:", input_element.get('value'))
            
            # Find and print all select elements within the form
            select_elements = form.find_all('select')
            if select_elements:
                print("  Select elements:")
                for select_element in select_elements:
                    print("    Name:", select_element.get('name'))
                    print("    Options:")
                    options = select_element.find_all('option')
                    for option in options:
                        print("      Value:", option.get('value'))
                        print("      Text:", option.text)
            
            # Find and print all textarea elements within the form
            textarea_elements = form.find_all('textarea')
            if textarea_elements:
                print("  Textarea elements:")
                for textarea_element in textarea_elements:
                    print("    Name:", textarea_element.get('name'))
                    print("    Value:", textarea_element.text)
            
            print()  # Add a new line between forms

    def display_page_content(self,page_content):        
        # Convert page content to HTML string
        html_content = str(page_content)
        
        # Create a temporary HTML file
        with open("temp.html","w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Open the temporary HTML file in the default web browser
        webbrowser.open("temp.html")

    
    def fill_and_submit_form(self,url, input_text):
        browser = self.browser
        browser.open(url)
        
        soup = browser.get_current_page()
        
        # Find the form named 'myform'
        form = browser.select_form('form[id="searchbox_homepage"]')
        
        # Find the input field named 'q' and fill it with input_text
        form.input({'q': input_text})
        
        # Submit the form
        browser.submit_selected()
        
        return browser.get_current_page()    
    
    def get_content_without_html(self,url, n):
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        soup = browser.get_current_page()
        
        # Find the first 'n' links on the page
        links = soup.find_all('a')[:n]
        
        linked_page_contents = []
        
        for link in links:
            # Follow each link
            browser.follow_link(link.get('href'))
            
            # Get the content of the linked page without HTML tags
            linked_page_content = browser.get_current_page().get_text()
            linked_page_contents.append(linked_page_content)
            
            # Go back to the original page for the next link
            #browser.back()
        
        return linked_page_contents

# Example usage
url ='https://duckduckgo.com/'
#url = "https://www.fruitssecsduweb.com/11-pistache"  # Change this URL to the desired one
num_links_to_follow = 3  # Change this number to the desired number of links to follow

b = Browse()
currentpage = b.debug_form_elements(url)

#content_without_html = get_content_without_html(url, num_links_to_follow)

#for content in content_without_html:
#    print(content)

import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:84.0) Gecko/20100101 Firefox/84.0",
}

print('going')
page = requests.get('https://duckduckgo.com/html/?q=test', headers=headers).text
soup = BeautifulSoup(page, 'html.parser').find_all("a", class_="result__url", href=True)

for link in soup:
    print(link['href'])