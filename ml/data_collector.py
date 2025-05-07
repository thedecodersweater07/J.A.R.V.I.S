import asyncio
import pandas as pd
from typing import List, Dict
import logging
from pathlib import Path
from datetime import datetime

from utils.data_scraper import DataScraper
from ml.data_processor import MLDataProcessor

logger = logging.getLogger(__name__)

class MLDataCollector:
    def __init__(self, output_dir: str = "data/ml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scraper = DataScraper()
        self.processor = MLDataProcessor()
        
    async def collect_and_process(self, sources: List[str], output_filename: str):
        """Collect data from sources and save to CSV"""
        # Scrape data
        raw_data = await self.scraper.scrape_data(sources)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'url': item['url'],
            'title': item['data']['title'],
            'content': item['data']['text'],
            'links': len(item['data']['links']),
            'timestamp': item['timestamp']
        } for item in raw_data if item])
        
        # Process data
        processed_df = self.processor.process_data(df)
        
        # Save to CSV
        output_path = self.output_dir / f"{output_filename}_{datetime.now().strftime('%Y%m%d')}.csv"
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return processed_df

    def merge_datasets(self, pattern: str = "*.csv") -> pd.DataFrame:
        """Merge all CSV files in the output directory"""
        all_data = []
        for csv_file in self.output_dir.glob(pattern):
            df = pd.read_csv(csv_file)
            all_data.append(df)
            
        merged_df = pd.concat(all_data, ignore_index=True)
        return merged_df
