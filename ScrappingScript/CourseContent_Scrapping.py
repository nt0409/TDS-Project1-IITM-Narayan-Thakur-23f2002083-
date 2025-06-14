from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import os
import time

TARGET_URL = "https://tds.s-anand.net/#/2025-01/"

def initialize_browser():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=chrome_options)

def wait_for_page(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(3)  # Additional buffer for JS rendering

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    selected_tags = soup.select('.content, .main-content, article, section, div, p, h1, h2, h3, li')
    extracted = " ".join(
        tag.get_text(strip=True) for tag in selected_tags if tag.get_text(strip=True)
    )
    if not extracted and soup.body:
        print("Fallback: No main tags matched. Using <body> content.")
        extracted = soup.body.get_text(separator=" ", strip=True)
    return extracted

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def store_data_as_json(content, output_file, source_url):
    payload = [{"text": content, "url": source_url}]
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f" Content saved at: {output_file}")
    except Exception as err:
        print(f"Failed to save content: {err}")
    return payload

def fetch_course_page_content():
    browser = initialize_browser()
    try:
        browser.get(TARGET_URL)
        wait_for_page(browser)
        html = browser.page_source
        text_content = extract_text_from_html(html)

        print(f" Content length: {len(text_content)} characters")
        if len(text_content) < 50:
            print("Very short content. Possible render issue.")
            print(" Page preview:", html[:300])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "..", "data")
        ensure_directory_exists(save_dir)
        output_path = os.path.join(save_dir, "course_content.json")

        return store_data_as_json(text_content, output_path, TARGET_URL)
    finally:
        browser.quit()

if __name__ == "__main__":
    fetch_course_page_content()
