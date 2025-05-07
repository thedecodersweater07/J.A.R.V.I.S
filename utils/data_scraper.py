import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict
from bs4 import BeautifulSoup
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataScraper:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_concurrent': 5,
            'timeout': 30,
            'retry_attempts': 3
        }
        
    async def scrape_data(self, urls: List[str]) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if r is not None]
    
    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Dict:
        try:
            async with session.get(url, timeout=self.config['timeout']) as response:
                if response.status == 200:
                    text = await response.text()
                    data = self._parse_content(text)
                    return {
                        'url': url,
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
        return None

    def _parse_content(self, html: str) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')
        return {
            'title': soup.title.string if soup.title else '',
            'text': ' '.join([p.text for p in soup.find_all('p')]),
            'links': [a['href'] for a in soup.find_all('a', href=True)]
        }
