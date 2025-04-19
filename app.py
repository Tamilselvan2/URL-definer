from flask import Flask, render_template, request
import urllib.parse
import re
import requests
import validators
from google import generativeai as genai

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyAExghwYHuQP_qkKJ50hrJFyAGKwjy0R34")  # Replace with your API key

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

def follow_redirects(url):
    """Follow URL redirects to get the final destination URL."""
    try:
        response = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        return response.url
    except:
        return url

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

    # YouTube link simplification
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

    # Amazon link simplification
    if "amazon." in domain:
        match = re.search(r'/dp/([A-Z0-9]{10})', path)
        if match:
            return f"https://{parsed.netloc}/dp/{match.group(1)}"

    # Flipkart link simplification
    if "flipkart.com" in domain:
        match = re.search(r'(/[^/]+/p/it[^\?/]+)', path)
        if match:
            return f"https://{parsed.netloc}{match.group(1)}"

    # Instagram link simplification
    if "instagram.com" in domain:
        # Profile link: /username/
        profile_match = re.match(r'^/([^/]+)/?$', path)
        if profile_match:
            username = profile_match.group(1)
            return f"https://instagram.com/{username}/"
        # Post link: /p/post_id/ or /reel/post_id/
        post_match = re.match(r'^/(p|reel)/([^/]+)/?$', path)
        if post_match:
            post_type = post_match.group(1)
            post_id = post_match.group(2)
            return f"https://instagram.com/{post_type}/{post_id}/"

    # Twitter (X) link simplification
    if "twitter.com" in domain or "x.com" in domain:
        # Profile link: /username
        profile_match = re.match(r'^/([^/]+)$', path)
        if profile_match:
            username = profile_match.group(1)
            return f"https://twitter.com/{username}"
        # Post link: /username/status/post_id
        post_match = re.match(r'^/([^/]+)/status/(\d+)$', path)
        if post_match:
            username = post_match.group(1)
            post_id = post_match.group(2)
            return f"https://twitter.com/{username}/status/{post_id}"

    # Fallback: remove query and fragment for other domains
    return urllib.parse.urlunparse(parsed._replace(query='', fragment=''))

def generate_summary(url):
    """Generate a summary of the webpage content using Google Generative AI."""
   
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    # Check if the URL is a YouTube link
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
        return response.text.strip()
    except Exception as e:
        return f"Summary error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    summary = None
    if request.method == "POST":
        input_url = request.form["url"]
        # Follow redirects and clean the URL
        final_url = follow_redirects(input_url)
        cleaned_url = remove_tracking_params(final_url)
        simplified_url = simplify_known_paths(cleaned_url)

        # Validate URL format and check for spam (unchanged)
        valid = validators.url(input_url)
        spam = any(x in input_url for x in ["spammy", "clickbait", "malware"])

        # Parse original URL for details (unchanged)
        parsed = urllib.parse.urlparse(input_url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query))
        trackers = {k: v for k, v in query_params.items() if k.lower() in TRACKERS}

        # Generate summary using the clean link
        summary = generate_summary(cleaned_url)

        # Prepare result dictionary
        result = {
            "original": input_url,
            "cleaned": simplified_url,  # Updated to use simplified URL
            "valid": valid,
            "spam": spam,
            "ml_label": "Safe" if not spam else "Spam",
            "ml_prob": 97 if not spam else 85,
            "protocol": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "query_params": query_params,
            "trackers": trackers,
        }

    return render_template("index.html", result=result, summary=summary)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
