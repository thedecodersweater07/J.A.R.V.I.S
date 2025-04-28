# parser.py
# Een module voor het parseren en verwerken van verschillende dataformaten

import json
import re
import csv
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import io

# Configuratie voor logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Parser")

class Parser:
    """
    Klasse voor het parseren van verschillende dataformaten en datastructuren.
    Ondersteunt JSON, CSV, XML, en custom formaten.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialiseert de Parser met optionele configuratie.
        
        Args:
            config: Een dictionary met configuratie-instellingen
        """
        self.config = config or {}
        self.parsers = {
            'json': self._parse_json,
            'csv': self._parse_csv,
            'xml': self._parse_xml,
            'text': self._parse_text
        }
        logger.info("Parser geïnitialiseerd met %d beschikbare parsers", len(self.parsers))
    
    def parse(self, content: str, format_type: str = None) -> Any:
        """
        Parst de gegeven content naar een Python-object.
        
        Args:
            content: De te parseren tekst
            format_type: Het type format ('json', 'csv', 'xml', 'text')
                         Als niet opgegeven, wordt het formaat automatisch gedetecteerd
                         
        Returns:
            Geparseerde data in het juiste Python-formaat
        """
        if not content:
            logger.warning("Lege content ontvangen om te parseren")
            return None
            
        # Detecteer format als het niet is opgegeven
        if not format_type:
            format_type = self._detect_format(content)
            logger.info("Format gedetecteerd als: %s", format_type)
        
        # Controleer of we de parser hebben
        if format_type not in self.parsers:
            logger.error("Geen parser beschikbaar voor format: %s", format_type)
            raise ValueError(f"Niet-ondersteund formaat: {format_type}")
        
        # Parse de content
        try:
            result = self.parsers[format_type](content)
            return result
        except Exception as e:
            logger.error("Fout bij parseren van %s: %s", format_type, str(e))
            raise ValueError(f"Fout bij parseren van {format_type}: {str(e)}")
    
    def _detect_format(self, content: str) -> str:
        """Detecteert automatisch het formaat van de content."""
        content = content.strip()
        
        # Controleer JSON
        if (content.startswith('{') and content.endswith('}')) or \
           (content.startswith('[') and content.endswith(']')):
            return 'json'
        
        # Controleer XML
        if content.startswith('<') and content.endswith('>'):
            return 'xml'
        
        # Controleer CSV (eenvoudige heuristiek)
        if ',' in content and '\n' in content:
            return 'csv'
        
        # Standaard naar text
        return 'text'
    
    def _parse_json(self, content: str) -> Any:
        """Parst JSON-content naar een Python-object."""
        return json.loads(content)
    
    def _parse_csv(self, content: str) -> List[Dict[str, str]]:
        """Parst CSV-content naar een lijst van dictionaries."""
        result = []
        with io.StringIO(content) as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append(dict(row))
        return result
    
    def _parse_xml(self, content: str) -> Dict[str, Any]:
        """Parst XML-content naar een dictionary."""
        root = ET.fromstring(content)
        return self._xml_element_to_dict(root)
    
    def _xml_element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Converteer XML-element naar dictionary."""
        result = {}
        
        # Verwerk attributen
        for key, value in element.attrib.items():
            result[f"@{key}"] = value
            
        # Verwerk child elementen
        for child in element:
            child_data = self._xml_element_to_dict(child)
            
            if child.tag in result:
                # Als we al een tag hebben, maak er een lijst van
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
                
        # Verwerk tekst
        text = element.text
        if text and text.strip():
            if len(result) == 0:
                # Als er geen child elementen of attributen zijn
                return text.strip()
            else:
                # Anders, sla de tekst op met een speciale key
                result["#text"] = text.strip()
                
        return result
    
    def _parse_text(self, content: str) -> str:
        """Parst platte tekst (geen speciale verwerking)."""
        return content