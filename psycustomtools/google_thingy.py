import re
import os
import mechanicalsoup


# Connect to duckduckgo
browser = mechanicalsoup.StatefulBrowser()
resp = browser.open("https://www.fruitssecsduweb.com/11-pistache")
#print(resp._content)

a = browser.get_current_page()
browser.select_form()
browser.get_current_form().print_summary()
page = browser.get_current_page()
links = page.find_all('a')
for link in links:
    print(f'\n{link}')
browser["q"] = "wimbledon"
#browser.launch_browser()

response = browser.submit_selected()
print(response)
new_url = browser.get_url()
print(new_url)
#resp = browser.open(new_url)
print(resp)

page = browser.get_current_page()
browser.select_form()
browser.get_current_form().print_summary()
links = page.find_all('a')
print(links)
os.abort()


forms = browser.get_current_page().find_all('form')
for aform in forms:
   print(aform)

# Fill-in the search form
browser.select_form('#search_form.search')
browser["q"] = "MechanicalSoup"
browser.submit_selected()

# Display the results
for link in browser.page.select('a.result__a'):
    print(link.text, '->', link.attrs['href'])