import logging
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from datetime import datetime, timedelta
import sqlite3
import json
from config.database import DATABASE_PATHS

logger = logging.getLogger(__name__)

class ExternalDataIntegrator:
    """Handles integration with external data sources and APIs."""
    
    def __init__(self, api_config: Dict[str, Any] = None):
        self.api_config = api_config or {}
        self.cache = {}
        self.cache_expiry = {}
        self.session = None
        self.db_path = DATABASE_PATHS["cache"]
        self._init_db()
        
    async def initialize(self):
        """Initialize async HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close async HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_data(self, source: str, query: Dict[str, Any], cache_ttl: int = 3600) -> Optional[Dict]:
        """Fetch data from external source with caching."""
        cache_key = f"{source}:{str(query)}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if source not in self.api_config:
            logger.error(f"Unknown data source: {source}")
            return None

        try:
            data = await self._make_request(source, query)
            if data:
                self._update_cache(cache_key, data, cache_ttl)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data from {source}: {e}")
            return None

    async def _make_request(self, source: str, query: Dict[str, Any]) -> Optional[Dict]:
        """Make HTTP request to external API."""
        config = self.api_config[source]
        url = config['url']
        headers = config.get('headers', {})
        
        if not self.session:
            await self.initialize()

        try:
            async with self.session.request(
                method=config.get('method', 'GET'),
                url=url,
                headers=headers,
                params=query if config.get('method') == 'GET' else None,
                json=query if config.get('method') == 'POST' else None,
                timeout=config.get('timeout', 30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache or key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]

    def _update_cache(self, key: str, data: Dict, ttl: int):
        """Update cache with new data and persist to database."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO api_cache (cache_key, data, expiry, source)
            VALUES (?, ?, ?, ?)
            ''', (key, json.dumps(data), self.cache_expiry[key].timestamp(), 
                  key.split(':')[0]))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist cache to database: {e}")

    def _init_db(self):
        """Initialize the cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT,
            expiry REAL,
            source TEXT
        )''')
        
        conn.commit()
        conn.close()

    def clear_cache(self, source: Optional[str] = None):
        """Clear cache for specific source or all sources."""
        if source:
            keys = [k for k in self.cache.keys() if k.startswith(f"{source}:")]
            for k in keys:
                self.cache.pop(k, None)
                self.cache_expiry.pop(k, None)
        else:
            self.cache.clear()
            self.cache_expiry.clear()
