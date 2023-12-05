# Selenium #KIJIJI
# https://sites.google.com/a/chromium.org/chromedreiver/downloads

from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
import time
import json

PATH = 'C://Users//sergi//ChromeDriver//chromedriver.exe'
driver = webdriver.Chrome(PATH)

driver.get("https://www.kijiji.ca/b-appartement-condo/ville-de-quebec/c37l1700124")

html_code = driver.page_source

from bs4 import BeautifulSoup

soup = BeautifulSoup(html_code, 'lxml')

all_house_li = soup.find_all("div", class_ = "search-item")

house_li = all_house_li[0]

def house_li_html_to_obj(house_li_html):
    price = house_li_html.find(class_="price").text
    price.replace("\n", "")
    print(price)

    title = house_li_html.find(class_="title").text
    title.replace("\n", "")
    print(title)

    location = house_li_html.find(class_="location").text
    print(location)

    date_posted = house_li_html.find(class_="date-posted").text
    print(date_posted)

    description = house_li_html.find(class_="description").text
    print(description)

    rental_info = house_li_html.find(class_="rental-info").text
    print(rental_info)

    image_URL = house_li_html.find(class_="image").text
    print(image_URL)

    return {"price": price, "title": title, "location": location, "date_posted": date_posted, "description": description, "rental_info": rental_info, "image_URL": image_URL}

file_path='kijiji.json'
print(file_path)
with open(file_path, 'w') as outfile:
    print("writing file to" ,file_path)
    time.sleep(5)
    for house_li in all_house_li:        
        json.dump(house_li_html_to_obj(house_li), outfile)
outfile.close()
print("Done")
#print(house_li_html_to_obj(house_li))