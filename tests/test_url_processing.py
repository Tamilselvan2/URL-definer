"""
Unit tests for URL processing functions.
Run with: python -m pytest tests/
"""

import unittest
import sys
import os
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    remove_tracking_params,
    simplify_known_paths,
    extract_handcrafted_features,
    SUSPICIOUS_KEYWORDS,
    TRACKERS
)


class TestRemoveTrackingParams(unittest.TestCase):
    """Test tracking parameter removal."""
    
    def test_remove_utm_params(self):
        """Test removal of UTM parameters."""
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&utm_campaign=test"
        result = remove_tracking_params(url)
        self.assertNotIn("utm_source", result)
        self.assertNotIn("utm_medium", result)
        self.assertNotIn("utm_campaign", result)
    
    def test_remove_fbclid(self):
        """Test removal of Facebook click ID."""
        url = "https://example.com/page?fbclid=12345&param=value"
        result = remove_tracking_params(url)
        self.assertNotIn("fbclid", result)
        self.assertIn("param=value", result)
    
    def test_preserve_legit_params(self):
        """Test that legitimate parameters are preserved."""
        url = "https://example.com/page?id=123&name=test"
        result = remove_tracking_params(url)
        self.assertIn("id=123", result)
        self.assertIn("name=test", result)
    
    def test_remove_fragment(self):
        """Test removal of URL fragments."""
        url = "https://example.com/page?param=value#section"
        result = remove_tracking_params(url)
        self.assertNotIn("#", result)
    
    def test_empty_query_string(self):
        """Test handling of URLs without query strings."""
        url = "https://example.com/page"
        result = remove_tracking_params(url)
        self.assertEqual(result, "https://example.com/page")


class TestSimplifyKnownPaths(unittest.TestCase):
    """Test URL simplification for known domains."""
    
    def test_youtube_video_id(self):
        """Test simplification of YouTube URLs with video ID."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=30"
        result = simplify_known_paths(url)
        self.assertIn("youtube.com", result)
        self.assertIn("v=dQw4w9WgXcQ", result)
    
    def test_youtube_playlist(self):
        """Test simplification of YouTube playlist URLs."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx&index=1"
        result = simplify_known_paths(url)
        self.assertIn("list=", result)
    
    def test_youtu_be_short(self):
        """Test simplification of youtu.be short links."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = simplify_known_paths(url)
        self.assertIn("youtube.com", result)
        self.assertIn("v=dQw4w9WgXcQ", result)
    
    def test_amazon_asin(self):
        """Test simplification of Amazon URLs to just ASIN."""
        url = "https://www.amazon.com/Laptop-Intel-i7-8GB-RAM/dp/B0123456789?ref=sr_1_1"
        result = simplify_known_paths(url)
        self.assertIn("B0123456789", result)
    
    def test_instagram_profile(self):
        """Test simplification of Instagram profile URLs."""
        url = "https://www.instagram.com/username/?hl=en"
        result = simplify_known_paths(url)
        self.assertIn("username", result)
    
    def test_twitter_profile(self):
        """Test simplification of Twitter profile URLs."""
        url = "https://twitter.com/username?lang=en"
        result = simplify_known_paths(url)
        self.assertIn("twitter.com/username", result)
    
    def test_twitter_x_profile(self):
        """Test simplification of X (Twitter) profile URLs."""
        url = "https://x.com/username?lang=en"
        result = simplify_known_paths(url)
        self.assertIn("twitter.com/username", result)
    
    def test_generic_url_cleanup(self):
        """Test cleanup of generic URLs (remove query and fragment)."""
        url = "https://example.com/page?ref=1&utm=value#section"
        result = simplify_known_paths(url)
        self.assertEqual(result, "https://example.com/page")


class TestExtractHandcraftedFeatures(unittest.TestCase):
    """Test handcrafted feature extraction."""
    
    def test_basic_feature_extraction(self):
        """Test basic feature extraction."""
        urls = ["https://example.com/page"]
        features = extract_handcrafted_features(urls)
        
        # Should return array with 5 features: [length, params, suspicious, subdomains, is_https]
        self.assertEqual(features.shape[1], 5)
        self.assertGreater(features[0][0], 0)  # URL length should be > 0
        self.assertEqual(features[0][4], 1)  # Should have HTTPS
    
    def test_suspicious_keyword_detection(self):
        """Test detection of suspicious keywords."""
        urls = ["https://example.com/login?verify=true&secure=yes"]
        features = extract_handcrafted_features(urls)
        
        # Feature at index 2 is has_suspicious
        self.assertEqual(features[0][2], 1)  # Should detect suspicious keywords
    
    def test_query_parameter_count(self):
        """Test query parameter counting."""
        urls = ["https://example.com/page?p1=v1&p2=v2&p3=v3"]
        features = extract_handcrafted_features(urls)
        
        # Feature at index 1 is num_query_params
        self.assertEqual(features[0][1], 3)  # Should count 3 params
    
    def test_http_vs_https(self):
        """Test HTTPS detection."""
        urls_http = ["http://example.com"]
        urls_https = ["https://example.com"]
        
        features_http = extract_handcrafted_features(urls_http)
        features_https = extract_handcrafted_features(urls_https)
        
        self.assertEqual(features_http[0][4], 0)  # HTTP = 0
        self.assertEqual(features_https[0][4], 1)  # HTTPS = 1
    
    def test_multiple_urls(self):
        """Test feature extraction for multiple URLs."""
        urls = [
            "https://example1.com/page",
            "https://example2.com/page?param=value",
            "http://example3.com/page"
        ]
        features = extract_handcrafted_features(urls)
        
        self.assertEqual(features.shape[0], 3)  # Should have 3 rows
        self.assertEqual(features.shape[1], 5)  # Should have 5 features


class TestTrackerConstant(unittest.TestCase):
    """Test the TRACKERS constant."""
    
    def test_trackers_is_set(self):
        """Test that TRACKERS set is not empty."""
        self.assertGreater(len(TRACKERS), 0)
    
    def test_common_trackers_present(self):
        """Test that common trackers are in the set."""
        self.assertIn('utm_source', TRACKERS)
        self.assertIn('fbclid', TRACKERS)
        self.assertIn('gclid', TRACKERS)


if __name__ == '__main__':
    unittest.main()
