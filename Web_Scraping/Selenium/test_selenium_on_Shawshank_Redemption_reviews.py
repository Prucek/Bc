#! /usr/bin/env python3

import os
import time
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.binary_location = '/usr/bin/google-chrome-stable'

chrome_driver = '/usr/bin/chromedriver'
#chromedriver = "./chromedriver"

os.environ["webdriver.chrome.driver"] = chrome_driver
driver = webdriver.Chrome(executable_path=chrome_driver, options=chrome_options)
thing_url = "https://www.imdb.com/title/tt0111161/reviews"
driver.get(thing_url)

lm = 0
while True:
    try:
        loadMoreButton = driver.find_element_by_id('load-more-trigger')
        time.sleep(5)
        loadMoreButton.click()
        lm += 1
        print(f'load more {lm}', file = sys.stderr)
        time.sleep(5)
    except Exception as e:
        print(e)
        break
print(driver.page_source)
driver.quit()
