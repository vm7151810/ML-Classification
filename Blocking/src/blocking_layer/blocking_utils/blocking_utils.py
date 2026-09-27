from src.config import config
import logging

logger = logging.getLogger(__name__)

def to_char_3grams(text: str) -> list[str]:
    """
    Generate character 3-grams for a given string.
    
    Args:
        text (str): The input string to tokenize.
        
    Returns:
        list[str]: A list of 3-character overlapping n-grams, or specific 
                   fallbacks for edge cases.
    """
    try:
        if not isinstance(text, str):
            return []
            
        # Strip and lower for Defense 2 check
        clean_text = text.strip().lower()
            
        # Fetch prefixes from config
        nullname_prefix = config.NULLNAME_PREFIX.strip().lower()
        nulladdr_prefix = config.NULLADDR_PREFIX.strip().lower()

        # Defense 1: Defensive fallback for empty strings
        if not text:
            return []
            
        # Defense 2: Explicit interception of 'nullname/nulladdr'
        if clean_text in (nullname_prefix, nulladdr_prefix):
            return []
            
        # Defense 3: Fallback for short queries
        if len(text) < 3:
            return [text]
            
        # REQ-BUTIL-02: Overlapping 3-character substrings
        return [text[i:i+3] for i in range(len(text) - 2)]
        
    except Exception as e:
        logger.error(f"Error generating 3-grams for input '{text}': {e}")
        return []
