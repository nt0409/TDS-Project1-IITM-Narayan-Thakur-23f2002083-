import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

COURSE_URL = "https://tds.s-anand.net/#/2025-01/"

def launch_headless_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=options)

def wait_until_page_loads(driver, delay=15):
    WebDriverWait(driver, delay).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(2.5)  # Buffer for JS-heavy pages

def parse_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    target_elements = soup.select('.content, .main-content, article, section, div, p, h1, h2, h3, li')

    combined_text = " ".join(
        element.get_text(strip=True) for element in target_elements if element.get_text(strip=True)
    )

    if not combined_text and soup.body:
        print("Fallback to full body content.")
        combined_text = soup.body.get_text(separator=" ", strip=True)

    return combined_text

def make_sure_folder_exists(path):
    os.makedirs(path, exist_ok=True)

def save_to_json(text, filepath, source_url):
    payload = [{"text": text, "url": source_url}]
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Saved content to: {filepath}")
    except Exception as e:
        print(f" Failed to save: {e}")
    return payload

def scrape_course_content():
    driver = launch_headless_browser()
    try:
        driver.get(COURSE_URL)
        wait_until_page_loads(driver)
        html_content = driver.page_source
        extracted_text = parse_visible_text(html_content)

        print(f" Extracted {len(extracted_text)} characters")
        if len(extracted_text) < 100:
            print(" Warning: Extracted content is very short.")

        base_path = os.path.abspath(os.path.join(__file__, "..", "..", "data"))
        make_sure_folder_exists(base_path)
        output_file = os.path.join(base_path, "CourseContentData.json")

        return save_to_json(extracted_text, output_file, COURSE_URL)
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_course_content()
