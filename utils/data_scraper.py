import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import os
import json
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataScraper:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_concurrent': 5,
            'timeout': 20,
            'retry_attempts': 3
        }
        self.output_dir = 'data/scraped_data'
        os.makedirs(self.output_dir, exist_ok=True)

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch content from a single URL."""
        for attempt in range(self.config['retry_attempts']):
            try:
                async with session.get(url, timeout=self.config['timeout']) as response:
                    if response.status == 200:
                        logger.info(f"Fetched: {url}")
                        return await response.text()
                    else:
                        logger.warning(f"{url} -> Status {response.status}")
            except Exception as e:
                logger.error(f"Error on {url} (Attempt {attempt+1}): {e}")
        return ""

    async def scrape_data(self, urls: List[str]) -> List[str]:
        """Scrape content from multiple URLs."""
        connector = aiohttp.TCPConnector(limit=self.config['max_concurrent'])
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r.strip()]


class HTMLDataScraper(DataScraper):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.output_file = os.path.join(self.output_dir, 'scraped_data.jsonl')

    async def parse_html(self, html: str, url: str) -> Dict:
        """Extract title, links, text, keywords and summary."""
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(" ", strip=True)
        words = re.findall(r'\b\w{5,}\b', text.lower())  # filter short common words
        common_words = Counter(words).most_common(10)

        summary = ' '.join(text.split()[:100]) + "..." if len(text.split()) > 100 else text

        return {
            'url': url,
            'title': soup.title.string if soup.title and soup.title.string else '',
            'links': [a['href'] for a in soup.find_all('a', href=True)],
            'keywords': [word for word, _ in common_words],
            'summary': summary,
            'full_text': text,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def scrape_and_save(self, urls: List[str]) -> None:
        """Scrape, parse and save structured data."""
        raw_html_list = await self.scrape_data(urls)
        parsed_data = []
        for html, url in zip(raw_html_list, urls):
            if html:
                parsed = await self.parse_html(html, url)
                parsed_data.append(parsed)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in parsed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"{len(parsed_data)} pages scraped and saved to {self.output_file}")


class DataSummarizer:
    def __init__(self, input_file: str):
        self.input_file = input_file

    def summarize(self) -> pd.DataFrame:
        """Read and convert JSONL data to DataFrame."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)

    def save_summary(self, output_file: str) -> None:
        """Export DataFrame summary to CSV."""
        df = self.summarize()
        df = df[['url', 'title', 'keywords', 'summary', 'timestamp']]
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Summary CSV saved to {output_file}")


def run_scraper(urls: List[str], output_csv: str):
    scraper = HTMLDataScraper()
    asyncio.run(scraper.scrape_and_save(urls))

    summarizer = DataSummarizer(scraper.output_file)
    summarizer.save_summary(output_csv)





