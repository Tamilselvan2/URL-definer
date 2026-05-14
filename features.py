import numpy as np
from urllib.parse import urlparse, parse_qs
import logging

logger = logging.getLogger(__name__)

SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'verify', 'free', 'win', 'promo', 'gift', 'click',
    'update', 'account', 'confirm', 'deal', 'redeem', 'prize', 'join'
]

def extract_handcrafted_features(urls):
    features = []
    for url in urls:
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            url_length = len(url)
            num_query_params = len(query_params)
            has_suspicious = int(any(keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS))
            domain = parsed.netloc.lower().replace('www.', '')
            num_subdomains = len(domain.split('.')) - 2 if domain else 0
            is_https = int(parsed.scheme == 'https')
            
            features.append([url_length, num_query_params, has_suspicious, num_subdomains, is_https])
        except Exception as e:
            logger.warning(f"Error extracting features for URL '{url}': {str(e)}")
            features.append([0, 0, 0, 0, 0])
    
    return np.array(features)
