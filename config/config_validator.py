import json
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    import warnings
    warnings.warn("jsonschema not installed, validation will be limited")
from typing import Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigValidator:
    def __init__(self):
        self.schema_path = Path(__file__).parent / "schema"
        self.schemas = {}
        self._load_schemas()
        
    def _load_schemas(self):
        """Load all JSON schema files"""
        for schema_file in self.schema_path.glob("*_schema.json"):
            with open(schema_file) as f:
                self.schemas[schema_file.stem] = json.load(f)
    
    def validate(self, config: Dict[str, Any], schema_name: str) -> bool:
        """Validate config against schema"""
        try:
            if not JSONSCHEMA_AVAILABLE:
                logger.warning("jsonschema not available, skipping validation")
                return True
                
            schema = self.schemas.get(f"{schema_name}_config_schema")
            if not schema:
                logger.error(f"Schema {schema_name} not found")
                return False
                
            jsonschema.validate(instance=config, schema=schema)
            return True
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
