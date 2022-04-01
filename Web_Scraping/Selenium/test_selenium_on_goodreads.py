#! /usr/bin/env python3

import os
import time
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

chrome_options.binary_location = '/usr/bin/chromium-browser'

chrome_options.add_experimental_option('prefs',  {
    "download.default_directory": '/mnt/data/webscrap/goodreads/',
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
    }
)

chrome_driver = '/mnt/minerva1/nlp-3/goodreads/scripts/chromedriver.91.0.4472.101'
#chromedriver = "./chromedriver"

os.environ["webdriver.chrome.driver"] = chrome_driver
driver = webdriver.Chrome(executable_path=chrome_driver, options=chrome_options)
thing_url = 'https://www.goodreads.com/book/show/1.Harry_Potter_and_the_Half_Blood_Prince?csm_scope=&amp;hide_last_page=true&amp;language_code=en&amp;page=3'
driver.get(thing_url)
time.sleep(1)
print(driver.page_source)
driver.quit()
