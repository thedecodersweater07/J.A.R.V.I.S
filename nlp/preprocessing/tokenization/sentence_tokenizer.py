"""
Sentence Tokenizer Module

This module provides functionality for splitting text into sentences,
handling various edge cases and abbreviations.
"""

import re
import string
from typing import List, Set, Dict, Tuple, Optional, Pattern, Match
import unicodedata

class SentenceTokenizer:
    """Tokenizes text into sentences with support for various languages and edge cases."""
    
    def __init__(self, language: str = 'english'):
        """Initialize the sentence tokenizer.
        
        Args:
            language: Language code (e.g., 'english', 'dutch')
        """
        self.language = language.lower()
        self.abbreviations = self._load_abbreviations()
        self.punctuation = set('.!?')
        self.quotes = {'"', "'", '`', '\u201c', '\u201d', '\u2018', '\u2019'}
        self.brackets = {'(', '[', '{', '<', '>', '}', ']', ')'}
        self.regex_patterns = self._init_regex_patterns()
    
    def _load_abbreviations(self) -> Set[str]:
        """Load language-specific abbreviations.
        
        Returns:
            Set of abbreviations
        """
        # Common English abbreviations
        abbreviations = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'hon.', 'st.', 'jr.', 'sr.',
            'u.s.', 'u.k.', 'u.n.', 'e.u.', 'a.d.', 'b.c.', 'b.c.e.', 'c.e.', 'a.m.',
            'p.m.', 'etc.', 'e.g.', 'i.e.', 'vs.', 'viz.', 'cf.', 'c.f.', 'ex.', 'fig.',
            'no.', 'nos.', 'vol.', 'vols.', 'ch.', 'chaps.', 'sec.', 'secs.', 'fig.',
            'figs.', 'p.', 'pp.', 'para.', 'paras.', 'ed.', 'eds.', 'trans.', 'approx.',
            'appx.', 'appxs.', 'ca.', 'cent.', 'cent.', 'cf.', 'col.', 'cols.', 'comp.',
            'comps.', 'corp.', 'dept.', 'div.', 'edn.', 'e.g.', 'esp.', 'est.', 'et al.',
            'etc.', 'ex.', 'fig.', 'figs.', 'fl.', 'ft.', 'fwd.', 'hr.', 'ibid.', 'id.',
            'i.e.', 'illus.', 'inc.', 'intl.', 'intro.', 'l.', 'll.', 'ms.', 'mss.',
            'n.b.', 'n.d.', 'n.p.', 'n.s.', 'op.', 'p.', 'pp.', 'para.', 'pl.', 'pls.',
            'q.v.', 'r.', 'repr.', 'rev.', 'sc.', 'sec.', 'sect.', 'ser.', 'soc.',
            'supp.', 'suppl.', 's.v.', 'trans.', 'univ.', 'var.', 'viz.', 'vol.', 'vs.'
        }
        
        # Add language-specific abbreviations
        if self.language == 'dutch':
            abbreviations.update({
                'dhr.', 'mw.', 'mevr.', 'dr.', 'prof.', 'ing.', 'ir.', 'drs.', 'mr.',
                'mrs.', 'b.v.', 'bijv.', 'bijz.', 'blz.', 'ca.', 'd.w.z.', 'e.d.', 'e.k.',
                'e.v.', 'e.c.', 'enz.', 'evt.', 'i.p.v.', 'i.t.t.', 'm.a.w.', 'm.b.t.',
                'm.i.', 'm.i.v.', 'm.u.v.', 'm.b.h.v.', 'n.a.v.', 'n.a.w.', 'n.m.t.',
                'n.v.t.', 'o.a.', 'o.b.v.', 'o.g.v.', 'o.i.d.', 'o.m.', 'o.t.', 'o.v.v.',
                'p/a', 'p.p.', 'p.s.', 'resp.', 's.v.p.', 't.a.v.', 't.g.v.', 't.o.v.',
                't.w.', 'v.a.', 'v.a.n.', 'v.b.', 'v.chr.', 'v.d.', 'v.h.', 'v.l.n.r.',
                'v.o.f.', 'v.w.t.', 'z.o.z.', 'z.v.'
            })
        
        return abbreviations
    
    def _init_regex_patterns(self) -> Dict[str, Pattern]:
        """Initialize regex patterns for sentence tokenization.
        
        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {
            # Common sentence boundaries
            'sentence_end': re.compile(
                r'(?<![A-Z][a-z]\.)(?<=\S[.!?])(?=\s+[A-Z\u00C0-\u00DC])',
                re.UNICODE
            ),
            # Abbreviations followed by capital letters (not sentence boundaries)
            'abbrev': re.compile(
                r'(?:\b\p{L}\.){2,}\p{L}*\.?|\b\p{L}\.\p{L}\.(?:\p{L}\.)*|\b\p{L}+\.[\p{L}\p{N}]+\.?',
                re.UNICODE | re.IGNORECASE
            ),
            # Decimal numbers
            'decimal': re.compile(r'\d+\.\d+'),
            # Email addresses
            'email': re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
            # URLs
            'url': re.compile(r'https?://\S+|www\.\S+'),
            # Ellipsis and multiple punctuation
            'ellipsis': re.compile(r'\.{2,}'),
            # Quotations and brackets
            'quotes': re.compile(r'[\"\'\u201c\u201d\u2018\u2019]'),
            'brackets': re.compile(r'[\[\](){}<>]'),
            # Punctuation with possible quotes/brackets
            'punct': re.compile(r'[.!?]+')
        }
        
        return patterns
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        # Normalize unicode and strip whitespace
        text = unicodedata.normalize('NFKC', text).strip()
        
        # Replace problematic patterns with placeholders
        placeholders = {}
        
        # Replace URLs
        urls = list(self.regex_patterns['url'].finditer(text))
        for i, match in enumerate(urls):
            placeholder = f'__URL_{i}__'
            placeholders[placeholder] = match.group()
            text = text[:match.start()] + placeholder + text[match.end():]
        
        # Replace email addresses
        emails = list(self.regex_patterns['email'].finditer(text))
        for i, match in enumerate(emails):
            placeholder = f'__EMAIL_{i}__'
            placeholders[placeholder] = match.group()
            text = text[:match.start()] + placeholder + text[match.end():]
        
        # Split into sentences
        sentences = []
        start = 0
        
        for match in self.regex_patterns['sentence_end'].finditer(text):
            # Check if this is a false positive (e.g., abbreviation)
            potential_sentence = text[start:match.end()].strip()
            if not self._is_sentence_boundary(potential_sentence):
                continue
                
            # Add the sentence
            sentences.append(potential_sentence)
            start = match.end()
        
        # Add the last sentence
        if start < len(text):
            sentences.append(text[start:].strip())
        
        # Restore placeholders
        for i, sentence in enumerate(sentences):
            for placeholder, original in placeholders.items():
                if placeholder in sentence:
                    sentences[i] = sentence.replace(placeholder, original)
        
        return [s for s in sentences if s]
    
    def _is_sentence_boundary(self, text: str) -> bool:
        """Check if the given text ends with a valid sentence boundary.
        
        Args:
            text: Text to check
            
        Returns:
            True if this is a valid sentence boundary, False otherwise
        """
        if not text:
            return False
            
        # Get the last token (word or punctuation)
        tokens = text.split()
        if not tokens:
            return False
            
        last_token = tokens[-1]
        
        # Check if the last token ends with sentence-ending punctuation
        if not any(last_token.endswith(p) for p in self.punctuation):
            return False
            
        # Check for abbreviations
        if self._is_abbreviation(last_token):
            return False
            
        # Check for numbers (e.g., version numbers, IP addresses)
        if any(c.isdigit() for c in last_token):
            # Check for decimal numbers
            if '.' in last_token and all(c.isdigit() or c == '.' for c in last_token):
                return False
                
            # Check for IP addresses
            if last_token.count('.') == 3 and all(part.isdigit() for part in last_token.split('.')):
                return False
        
        return True
    
    def _is_abbreviation(self, token: str) -> bool:
        """Check if a token is an abbreviation.
        
        Args:
            token: Token to check
            
        Returns:
            True if the token is an abbreviation, False otherwise
        """
        # Remove any trailing punctuation
        clean_token = token.rstrip(''.join(self.punctuation))
        
        # Check if it's in our abbreviations set
        if clean_token.lower() in self.abbreviations:
            return True
            
        # Check for common abbreviation patterns
        if '.' in clean_token and clean_token[-1] != '.':
            parts = clean_token.split('.')
            if all(part.isalpha() and len(part) <= 2 for part in parts if part):
                return True
                
        # Check for single letters with periods (e.g., 'a.', 'b.')
        if len(clean_token) == 1 and clean_token.isalpha():
            return True
            
        return False
    
    def add_abbreviation(self, abbr: str) -> None:
        """Add an abbreviation to the list of known abbreviations.
        
        Args:
            abbr: Abbreviation to add
        """
        if abbr:
            self.abbreviations.add(abbr.lower())
    
    def add_abbreviations(self, abbrs: Set[str]) -> None:
        """Add multiple abbreviations to the list of known abbreviations.
        
        Args:
            abbrs: Set of abbreviations to add
        """
        if abbrs:
            self.abbreviations.update(abbr.lower() for abbr in abbrs if abbr)
    
    def set_language(self, language: str) -> None:
        """Set the language for sentence tokenization.
        
        Args:
            language: Language code (e.g., 'en', 'nl')
        """
        self.language = language.lower()
        self.abbreviations = self._load_abbreviations()
    
    def __call__(self, text: str) -> List[str]:
        """Alias for tokenize method to make the instance callable."""
        return self.tokenize(text)
