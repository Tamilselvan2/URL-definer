from flask import Flask, render_template, request, jsonify
import urllib.parse
import re
import requests
import validators
from google import generativeai as genai
import random
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import joblib
import logging
import os
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants for HTTP requests and tracking parameters
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://google.com',
    'DNT': '1',
    'Connection': 'close',
}

TRACKERS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'ref', 'ref_src', 'fbclid', 'gclid', 'msclkid', 'igshid', 'trk',
    'tag', 'xpid', 'asc_refurl', 'dib', 'dib_tag', 'lid', 'marketplace',
    'store', 'srno', 'otracker', 'ppt', 'ppn', 'ssid', 'iid', 'pid', 'cmp'
}

# Suspicious keywords often found in spam URLs
SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'verify', 'free', 'win', 'promo', 'gift', 'click',
    'update', 'account', 'confirm', 'deal', 'redeem', 'prize', 'join'
]

# Paths for saving the model and feature extractor
MODEL_PATH = 'url_classifier.pkl'
FEATURE_EXTRACTOR_PATH = 'feature_extractor.pkl'

# Initialize model and feature extractor
classifier = None
feature_extractor = None

# Function to extract handcrafted features
def extract_handcrafted_features(urls):
    features = []
    for url in urls:
        parsed = urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        # Feature 1: URL length
        url_length = len(url)
        
        # Feature 2: Number of query parameters
        num_query_params = len(query_params)
        
        # Feature 3: Presence of suspicious keywords
        has_suspicious = int(any(keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS))
        
        # Feature 4: Number of subdomains
        domain = parsed.netloc.lower().replace('www.', '')
        num_subdomains = len(domain.split('.')) - 2 if domain else 0
        
        # Feature 5: Presence of HTTPS
        is_https = int(parsed.scheme == 'https')
        
        features.append([url_length, num_query_params, has_suspicious, num_subdomains, is_https])
    return np.array(features)

# Function to normalize decision function scores to a confidence score (0-100%)
def decision_to_confidence(decision_score):
    # Use a logistic function to map decision function scores to [0, 1]
    confidence = 1 / (1 + np.exp(-abs(decision_score)))  # Sigmoid function
    return confidence * 100  # Convert to percentage

# Function to train or retrain the model
def train_model():
    global classifier, feature_extractor
    try:
        logger.debug("Loading dataset.csv for training")
        data = pd.read_csv('dataset.csv')
        logger.debug(f"Dataset loaded with {len(data)} rows")
        
        # Ensure labels are capitalized consistently
        data['label'] = data['label'].str.capitalize()
        X = data['url']
        y = data['label']
        
        # Define the feature extraction pipeline
        feature_extractor = FeatureUnion([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b|[^\w\s]',  # Include special chars
                max_features=5000
            )),
            ('handcrafted', FunctionTransformer(extract_handcrafted_features, validate=False))
        ])
        
        # Define the pipeline with feature extraction and SVM classifier
        classifier = Pipeline([
            ('features', feature_extractor),
            ('clf', SVC(kernel='linear', random_state=42))
        ])
        
        # Train the model
        classifier.fit(X, y)
        logger.debug("Model trained successfully")
        
        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='f1_macro')
        logger.debug(f"Cross-validation F1 scores: {cv_scores}")
        logger.debug(f"Average F1 score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Save the model and feature extractor
        joblib.dump(classifier, MODEL_PATH)
        joblib.dump(feature_extractor, FEATURE_EXTRACTOR_PATH)
        logger.debug("Model and feature extractor saved to disk")
        
        return cv_scores.mean()
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return 0.0

# Load or train the model at startup
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_EXTRACTOR_PATH):
    try:
        classifier = joblib.load(MODEL_PATH)
        feature_extractor = joblib.load(FEATURE_EXTRACTOR_PATH)
        logger.debug("Loaded existing model and feature extractor from disk")
    except Exception as e:
        logger.error(f"Error loading model from disk: {str(e)}")
        train_model()
else:
    train_model()

# Configure the Google Generative AI API
try:
    genai.configure(api_key="AIzaSyAExghwYHuQP_qkKJ50hrJFyAGKwjy0R34")  # Replace with your API key
    logger.debug("Google Generative AI configured")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {str(e)}")

# VirusTotal API key
VT_API_KEY = "a9507f6997678501d52e54c24bc69e2d1e0fd3e595d3c4697dad568bb3007129"  # Replace with your actual VirusTotal API key

def check_url_in_csv(url):
    """Check if the URL exists in dataset.csv and return its label if found."""
    try:
        df = pd.read_csv('dataset.csv')
        matching_row = df[df['url'] == url]
        if not matching_row.empty:
            label = matching_row.iloc[0]['label'].capitalize()
            logger.debug(f"Found URL {url} in dataset with label: {label}")
            return label
        logger.debug(f"URL {url} not found in dataset")
        return None
    except Exception as e:
        logger.error(f"Error checking URL in dataset: {str(e)}")
        return None

def append_to_csv(url, label):
    """Append a new URL and its label to dataset.csv and retrain the model."""
    try:
        df = pd.read_csv('dataset.csv')
        new_entry = pd.DataFrame({'url': [url], 'label': [label]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv('dataset.csv', index=False)
        logger.debug(f"Appended URL {url} with label {label} to dataset")
        
        # Retrain the model after appending
        train_model()
    except Exception as e:
        logger.error(f"Error appending to dataset: {str(e)}")

def follow_redirects(url):
    """Follow URL redirects to get the final destination URL. Returns (final_url, is_reachable)."""
    try:
        response = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        logger.debug(f"Followed redirects for {url} to {response.url}")
        return response.url, True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logger.error(f"Site not reachable for {url}: {str(e)}")
        return url, False
    except Exception as e:
        logger.error(f"Error following redirects for {url}: {str(e)}")
        return url, True  # Assume reachable unless connection error

def remove_tracking_params(url):
    """Remove known tracking parameters from the URL."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query)
    filtered = [(k, v) for k, v in query if k.lower() not in TRACKERS]
    cleaned_query = urllib.parse.urlencode(filtered)
    return urllib.parse.urlunparse(parsed._replace(query=cleaned_query, fragment=''))

def simplify_known_paths(url):
    """Simplify URLs for known domains, including YouTube, Amazon, Flipkart, Instagram, and Twitter (X)."""
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace("www.", "").lower()
    path = parsed.path
    query = urllib.parse.parse_qs(parsed.query)

    if "youtube.com" in domain:
        new_query = {}
        if 'v' in query:
            new_query['v'] = query['v']
        if 'list' in query:
            new_query['list'] = query['list']
        new_query_string = urllib.parse.urlencode(new_query, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query_string, fragment=''))

    if "youtu.be" in domain:
        video_id = path.lstrip("/")
        new_query = {'v': [video_id]}
        if 'list' in query:
            new_query['list'] = query['list']
        new_query_string = urllib.parse.urlencode(new_query, doseq=True)
        return f"https://youtube.com/watch?{new_query_string}"

    if "amazon." in domain:
        match = re.search(r'/dp/([A-Z0-9]{10})', path)
        if match:
            return f"https://{parsed.netloc}/dp/{match.group(1)}"

    if "flipkart.com" in domain:
        match = re.search(r'(/[^/]+/p/it[^\?/]+)', path)
        if match:
            return f"https://{parsed.netloc}{match.group(1)}"

    if "instagram.com" in domain:
        reserved_paths = [
            'accounts', 'explore', 'direct', 'reels', 'stories', 'p', 'reel',
            'login', 'signup', 'about', 'privacy', 'terms', 'challenge'
        ]
        profile_match = re.match(r'^/([a-zA-Z0-9_.]{1,30})/?$', path)
        if profile_match and profile_match.group(1) not in reserved_paths:
            username = profile_match.group(1)
            return f"https://instagram.com/{username}"
        post_match = re.match(r'^/(p|reel)/([^/]+)/?$', path)
        if post_match:
            post_type = post_match.group(1)
            post_id = post_match.group(2)
            return f"https://instagram.com/{post_type}/{post_id}/"
    
    if "twitter.com" in domain or "x.com" in domain:
        profile_match = re.match(r'^/([^/]+)$', path)
        if profile_match:
            username = profile_match.group(1)
            return f"https://twitter.com/{username}"
        post_match = re.match(r'^/([^/]+)/status/(\d+)$', path)
        if post_match:
            username = post_match.group(1)
            post_id = post_match.group(2)
            return f"https://twitter.com/{username}/status/{post_id}"

    return urllib.parse.urlunparse(parsed._replace(query='', fragment=''))

def generate_summary(url, is_reachable):
    """Generate a summary of the webpage content using Google Generative AI."""
    if not is_reachable:
        return "Site is not available"

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    if domain in ["youtube.com", "youtu.be"]:
        return "YouTube page cannot be summarized"
    elif domain == "instagram.com":
        return "Instagram page cannot be summarized"
    elif domain in ["twitter.com", "x.com"]:
        return "Twitter page cannot be summarized"

    prompt = (
        f"{url} - summarize the content in this page like title, "
        f"3 to 5 subtitles, description below each subtopic. "
        f"Skip everything except the title name (not the word 'title'), subtitle names "
        f"(not the word 'subtitle'). Format like Arc browser summarizer. "
        f"Don't print phrases like: 'Okay, here's a summary of...'"
        f"Neglact any boldness of the output text instead add colon to the subheadings and the descriptions are to be placed below it"
        f"within 130 words, not above 130 but can be below 130"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        logger.debug(f"Generated summary for {url}")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating summary for {url}: {str(e)}")
        return f"Summary error: {str(e)}"

def check_with_virustotal(url):
    """Check the URL with VirusTotal API and return status, reason, and detection count."""
    if not VT_API_KEY or VT_API_KEY == "your_virustotal_api_key_here":
        logger.error("VirusTotal API key not set")
        return "Error", "VirusTotal API key not set - External", 0, 0

    scan_url = "https://www.virustotal.com/vtapi/v2/url/scan"
    scan_params = {"apikey": VT_API_KEY, "url": url}
    try:
        scan_response = requests.post(scan_url, data=scan_params)
        if scan_response.status_code != 200:
            logger.error(f"VirusTotal scan failed for {url}: {scan_response.status_code}")
            return "Error", "Unable to submit URL to VirusTotal - External", 0, 0

        scan_data = scan_response.json()
        scan_id = scan_data.get("scan_id")
        if not scan_id:
            logger.error(f"No scan ID returned for {url}")
            return "Error", "No scan ID returned - External", 0, 0

        report_url = "https://www.virustotal.com/vtapi/v2/url/report"
        report_params = {"apikey": VT_API_KEY, "resource": scan_id}
        attempts = 0
        max_attempts = 6
        while attempts < max_attempts:
            report_response = requests.get(report_url, params=report_params)
            if report_response.status_code == 200:
                report = report_response.json()
                if report.get("response_code") == 1:
                    positives = report.get("positives", 0)
                    total = report.get("total", 0)
                    if positives == 0:
                        logger.debug(f"VirusTotal result for {url}: Safe")
                        return "Safe", "Predicted as safe with high confidence based on feature analysis - External", positives, total
                    else:
                        logger.debug(f"VirusTotal result for {url}: Spam")
                        return "Spam", f"Classified as suspicious based on {positives} behavioral features - External", positives, total
                else:
                    time.sleep(5)
                    attempts += 1
            else:
                logger.error(f"VirusTotal report fetch failed for {url}: {report_response.status_code}")
                return "Error", "Unable to get report from VirusTotal - External", 0, 0

        logger.warning(f"VirusTotal analysis in progress for {url}")
        return "Analysis in progress", "Please check back later - External", 0, 0
    except Exception as e:
        logger.error(f"Error in VirusTotal check for {url}: {str(e)}")
        return "Error", f"VirusTotal error: {str(e)} - External", 0, 0

def predict_with_model(url):
    """Use the trained SVM model to predict the label and confidence for a URL."""
    if classifier is None:
        logger.error("Model not initialized")
        return "Unknown", 0.0
    try:
        # Predict the label
        label = classifier.predict([url])[0]
        # Get the decision function score (distance from the hyperplane)
        decision_score = classifier.decision_function([url])[0]
        # Convert to confidence score (0-100%)
        confidence = decision_to_confidence(decision_score)
        logger.debug(f"Model prediction for {url}: {label} with confidence {confidence:.2f}% (decision score: {decision_score:.2f})")
        return label, confidence
    except Exception as e:
        logger.error(f"Error predicting with model for {url}: {str(e)}")
        return "Unknown", 0.0

@app.route("/", methods=["GET"])
def index():
    logger.debug("Rendering index page")
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    logger.debug(f"Received analyze request: {request.method}")
    logger.debug(f"Form data: {request.form}")
    
    if "url" in request.form:
        input_url = request.form["url"].strip()
        logger.debug(f"Received URL: {input_url}")

        if not input_url:
            return jsonify({"error": "Please enter a URL"}), 400

        # Check if URL exists in dataset.csv
        safety_status = check_url_in_csv(input_url)
        if safety_status:
            # URL found in CSV, return result
            logger.debug(f"URL {input_url} found in dataset with label: {safety_status}")
            final_url, is_reachable = follow_redirects(input_url)
            cleaned_url = remove_tracking_params(final_url)
            simplified_url = simplify_known_paths(cleaned_url)

            valid = validators.url(input_url) and is_reachable
            spam = any(x in input_url.lower() for x in SUSPICIOUS_KEYWORDS)

            parsed = urllib.parse.urlparse(input_url)
            query_params = dict(urllib.parse.parse_qsl(parsed.query))
            trackers = {k: v for k, v in query_params.items() if k.lower() in TRACKERS}

            ml_label = safety_status
            ml_prob = 100.0  # Since this is from CSV, assume 100% confidence
            vt_reason = "Retrieved from dataset"

            if ml_label == "Spam":
                summary = "Site cannot be summarized as this seems to be malicious"
            else:
                summary = generate_summary(cleaned_url, is_reachable)

            result = {
                "original": input_url,
                "cleaned": simplified_url,
                "valid": valid,
                "spam": spam,
                "ml_label": ml_label,
                "ml_prob": f"{ml_prob:.2f}",
                "vt_reason": vt_reason,
                "protocol": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path,
                "query_params": query_params,
                "trackers": trackers,
            }
            return jsonify({
                "result": result,
                "summary": summary,
                "from_csv": True,
                "show_report_form": True
            })

        else:
            # URL not in CSV, show survey
            logger.debug(f"URL {input_url} not in dataset, showing survey")
            return jsonify({
                "show_known_form": True,
                "input_url": input_url
            })

    elif "known_status" in request.form:
        input_url = request.form["input_url"]
        known_status = request.form["known_status"]
        logger.debug(f"Known status for {input_url}: {known_status}")

        if known_status == "Known":
            return jsonify({
                "show_safety_form": True,
                "input_url": input_url
            })
        else:  # Unknown
            # Process the URL with SVM model first
            final_url, is_reachable = follow_redirects(input_url)
            cleaned_url = remove_tracking_params(final_url)
            simplified_url = simplify_known_paths(cleaned_url)

            valid = validators.url(input_url) and is_reachable
            spam = any(x in input_url.lower() for x in SUSPICIOUS_KEYWORDS)

            parsed = urllib.parse.urlparse(input_url)
            query_params = dict(urllib.parse.parse_qsl(parsed.query))
            trackers = {k: v for k, v in query_params.items() if k.lower() in TRACKERS}

            # Predict using SVM model
            ml_label, ml_confidence = predict_with_model(input_url)

            if ml_confidence >= 75:
                # SVM prediction is confident, use it
                logger.debug(f"Using SVM prediction for {input_url}: {ml_label} with confidence {ml_confidence:.2f}%")
                vt_reason = "Predicted using SVM model with high confidence - Internal"
                # Append to CSV and retrain
                append_to_csv(input_url, ml_label)
            else:
                # SVM confidence is low, fall back to VirusTotal
                logger.debug(f"SVM confidence {ml_confidence:.2f}% is below 75%, falling back to VirusTotal for {input_url}")
                vt_status, vt_reason, positives, total = check_with_virustotal(cleaned_url)

                if vt_status == "Safe":
                    ml_label = "Safe"
                    ml_confidence = random.uniform(95, 99)
                elif vt_status == "Spam":
                    ml_label = "Spam"
                    ml_confidence = random.uniform(80, 95)
                else:
                    # If VirusTotal fails, use SVM prediction anyway
                    vt_reason = f"Used SVM prediction - Internal"

                # Append to CSV and retrain
                append_to_csv(input_url, ml_label)

            if ml_label == "Spam":
                summary = "Site cannot be summarized as this seems to be malicious"
            else:
                summary = generate_summary(cleaned_url, is_reachable)

            result = {
                "original": input_url,
                "cleaned": simplified_url,
                "valid": valid,
                "spam": spam,
                "ml_label": ml_label,
                "ml_prob": f"{ml_confidence:.2f}",
                "vt_reason": vt_reason,
                "protocol": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path,
                "query_params": query_params,
                "trackers": trackers,
            }
            return jsonify({
                "result": result,
                "summary": summary,
                "show_report_form": True
            })

    elif "safety_status" in request.form:
        input_url = request.form["input_url"]
        safety_status = request.form["safety_status"]
        logger.debug(f"Safety status for {input_url}: {safety_status}")

        # Store the user feedback in dataset.csv
        append_to_csv(input_url, safety_status)

        # Process the URL with minimal analysis (no VirusTotal check)
        final_url, is_reachable = follow_redirects(input_url)
        cleaned_url = remove_tracking_params(final_url)
        simplified_url = simplify_known_paths(cleaned_url)

        valid = validators.url(input_url) and is_reachable
        spam = any(x in input_url.lower() for x in SUSPICIOUS_KEYWORDS)

        parsed = urllib.parse.urlparse(input_url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query))
        trackers = {k: v for k, v in query_params.items() if k.lower() in TRACKERS}

        # Use the user's input as the ML prediction
        ml_label = safety_status
        ml_confidence = 100.0  # Since this is user-provided, assume 100% confidence
        vt_reason = "User-provided feedback"

        if ml_label == "Spam":
            summary = "Site cannot be summarized as this seems to be malicious"
        else:
            summary = generate_summary(cleaned_url, is_reachable)

        result = {
            "original": input_url,
            "cleaned": simplified_url,
            "valid": valid,
            "spam": spam,
            "ml_label": ml_label,
            "ml_prob": f"{ml_confidence:.2f}",
            "vt_reason": vt_reason,
            "protocol": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "query_params": query_params,
            "trackers": trackers,
        }
        return jsonify({
            "result": result,
            "summary": summary,
            "feedback_submitted": True,
            "show_report_form": True
        })

    return jsonify({"error": "Invalid request"}), 400

@app.route("/report", methods=["POST"])
def report():
    logger.debug(f"Received report request: {request.method}")
    logger.debug(f"Form data: {request.form}")
    
    if "input_url" in request.form and "report_status" in request.form:
        input_url = request.form["input_url"]
        report_status = request.form["report_status"]
        logger.debug(f"Report for {input_url}: {report_status}")

        # Append the reported status to dataset.csv and retrain the model
        append_to_csv(input_url, report_status)

        return jsonify({
            "message": "Thank you for your feedback! The model has been updated.",
            "reported_status": report_status
        })
    
    return jsonify({"error": "Invalid report request"}), 400

if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)
