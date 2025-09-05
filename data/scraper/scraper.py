import os
import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from urllib.parse import urljoin, urlparse
from fake_useragent import UserAgent
from retrying import retry
import re

BASE_URL = "https://en.wikipedia.org/wiki/"
RAW_TEXT_DIR = "data/raw/text"
os.makedirs(RAW_TEXT_DIR, exist_ok=True)
HEADERS = {"User-Agent": UserAgent().random}
logging.basicConfig(filename='scraper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PAGES = 1000
CRAWL_DELAY = 1
MAX_RETRIES = 3

class Scraper:
    def __init__(self, base_url=BASE_URL, raw_text_dir=RAW_TEXT_DIR):
        self.base_url = base_url
        self.raw_text_dir = raw_text_dir
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=2000)
    def fetch_page(self, url):
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    def extract_links(self, soup):
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/wiki/') and ':' not in href:  # alleen echte artikelen
                full_url = urljoin("https://en.wikipedia.org", href)
                if full_url not in self.visited_urls:
                    links.add(full_url)
        return links

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def save_text(self, title, text):
        filename = re.sub(r'[\\/*?:"<>|]', "_", title) + ".txt"
        filepath = os.path.join(self.raw_text_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Saved: {filepath}")

    def scrape(self, start_url, max_pages=MAX_PAGES):
        to_visit = [start_url]
        while to_visit and len(self.visited_urls) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            try:
                html = self.fetch_page(current_url)
                soup = BeautifulSoup(html, 'html.parser')
                title_tag = soup.find('h1')
                if not title_tag:
                    self.visited_urls.add(current_url)
                    continue
                title = title_tag.text
                paragraphs = soup.find_all('p')
                page_text = ' '.join([self.clean_text(p.get_text()) for p in paragraphs])
                if len(page_text.split()) < 50:  # skip heel korte pagina's
                    logging.info(f"Skipped (too short): {current_url}")
                    self.visited_urls.add(current_url)
                    continue
                self.save_text(title, page_text)
                self.visited_urls.add(current_url)
                links = self.extract_links(soup)
                to_visit.extend(links - self.visited_urls)
                time.sleep(CRAWL_DELAY + random.uniform(0, 1))
            except Exception as e:
                logging.error(f"Error fetching {current_url}: {e}")
        logging.info("Scraping completed.")

if __name__ == "__main__":
    scraper = Scraper()
    scraper.scrape(BASE_URL)
    print("Scraping completed. Check scraper.log for details.")
