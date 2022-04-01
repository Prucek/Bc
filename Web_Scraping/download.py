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

chrome_driver = '/mnt/minerva1/nlp-3/goodreads/scripts/chromedriver.97.0.4692.36'
#chrome_driver = 'home/xrucek00/chromedriver.97.0.4692.20'

os.environ["webdriver.chrome.driver"] = chrome_driver
driver = webdriver.Chrome(executable_path=chrome_driver, options=chrome_options)

file_name = sys.argv[1]
if not os.path.isfile(file_name):
    exit()


with open(file_name) as file:
    lines = file.readlines()
    for url in lines:
        new_file_name = url.split('/')[-1].rstrip("\r\n")
        try:
            driver.get(url)
        except:
            continue
        time.sleep(1)
        file= open(new_file_name,'a')
        file.write(driver.page_source)
        file.close()

driver.quit()
