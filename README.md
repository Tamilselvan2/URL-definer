# URL Definer

## Description

The URL Definer is a web-based tool designed to enhance user safety, privacy, and the readability of web links. It achieves this by cleaning, simplifying, and classifying URLs as either "Safe" or "Spam". The project is developed as a web application using the Flask framework for the backend and HTML, CSS, and JavaScript for the frontend. It's designed to be universally accessible and usable on both desktop and mobile devices without requiring installation.

## Main Functionalities

The URL Definer provides several key functionalities:

*   **URL Cleaning**: Removes unnecessary tracking parameters (e.g., `utm_source`, `fbclid`) and fragments from URLs to protect user privacy and create cleaner links.
*   **URL Simplification**: Shortens and clarifies URLs, particularly for known domains like YouTube and Amazon, making them more readable while retaining essential information.
*   **URL Classification**: Categorizes URLs as "Safe" or "Spam" using a hybrid approach that combines rule-based methods and a Support Vector Machine (SVM) machine learning model.
*   **URL Summary (Limited)**: Attempts to generate a summary of the URL's content. This functionality is noted as being limited for certain domains like social media sites.
*   **URL Validation**: Checks if the input URL is valid and reachable.

## How it Works

The system follows these general steps:

1.  **User Input**: The user enters a URL into the web form. Frontend JavaScript validates the input before sending it to the backend.
2.  **Backend Processing - Initial Check & Survey**:
    *   The backend checks if the URL exists in a local `dataset.csv`. If found, its stored label (Safe/Spam) is used.
    *   If not found, the user is surveyed about the site's legitimacy (Known/Unknown).
        *   If "Known," the user specifies if it's safe/unsafe, and this is added to `dataset.csv`, and the model is retrained.
        *   If "Unknown" (or if the URL was already in the dataset), automated analysis proceeds.
3.  **Backend Processing - URL Analysis Pipeline**:
    *   **Redirect Resolution**: Follows redirects to find the final destination URL using the `requests` library.
    *   **Tracking Parameter Removal**: Strips tracking parameters (e.g., `utm_source`, `fbclid`) using `urllib.parse`.
    *   **URL Simplification**: Applies rules and regular expressions to shorten URLs for known domains (YouTube, Amazon, etc.) or removes query parameters/fragments for others.
    *   **URL Classification (ML/VT)**:
        *   A trained Support Vector Machine (SVM) model predicts "Safe" or "Spam" based on TF-IDF features and handcrafted URL attributes (length, keywords, subdomains, HTTPS).
        *   If ML confidence is >75%, its prediction is used, and the URL/label is added to `dataset.csv` for periodic model retraining.
        *   If ML confidence is <75%, it falls back to the VirusTotal API (if configured) for external validation.
    *   **Rule-Based Classification**: A lightweight check for suspicious keywords (e.g., "free," "win," "login") flags potential spam.
    *   **Content Summarization**: Uses the Google Generative AI API (Gemini model) to summarize webpage content (excluding video/social media sites, which get a predefined message).
4.  **Result Display**:
    *   Analysis results (original URL, cleaned/simplified final URL, validity, spam status, ML/VirusTotal prediction, protocol, domain, path, query parameters) and the summary are sent to the frontend as JSON.
    *   The frontend dynamically displays these results with interactive elements.
5.  **User Reporting/Feedback**:
    *   Users can report if the site was actually safe or unsafe.
    *   This feedback updates the URL's label in `dataset.csv`, and the model is retrained.

## Technologies Used

*   **Frontend**: HTML5, CSS3, JavaScript
*   **Backend**: Flask (Python framework)
*   **URL Processing**: Python (`urllib.parse`, `re`)
*   **Machine Learning**: scikit-learn (SVM), joblib, pandas, numpy
*   **External Services (API Key Required)**: Google Generative AI (for summarization), VirusTotal (for safety checks)

## Redirect Handling

The URL Definer attempts to follow redirects for an input URL using the `requests` library to determine the final destination before analysis.

## User Feedback

Users can provide feedback on the accuracy of the "Safe"/"Spam" classification via a survey section. This feedback is appended to a local dataset (`dataset.csv`) and can be used to improve the classification model over time.

## Potential Future Enhancements

*   Implement full functionality of external API integrations (Google Generative AI, VirusTotal) by ensuring valid API keys are used.
*   Continuously update and expand the training dataset with more diverse and recent URLs.
*   Explore alternative or more sophisticated machine learning models for improved classification accuracy.
*   Add support for simplifying URLs from a wider range of popular websites.
*   Further enhance the user interface and experience based on user feedback.

## Installation and Usage

The URL Definer is a web-based application and does not require local installation. Users can access it through their web browser.

To run the project locally (for development):

1.  Clone the repository.
2.  Ensure Python and pip are installed.
3.  Install dependencies: `pip install -r requirements.txt`
4.  (Optional) Configure API keys for Google Generative AI and VirusTotal by setting the appropriate environment variables or modifying the configuration files. (Refer to project-specific documentation if available for exact environment variable names or config file paths).
5.  Run the Flask application: `python app.py`
6.  Access the application in your browser, typically at `http://127.0.0.1:5000/`.

## Contributing

(Placeholder: Details on how to contribute to the project, such as guidelines for pull requests, coding standards, etc., can be added here.)

Example:
We welcome contributions! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change. Ensure your code adheres to the project's coding standards.

## License

(Placeholder: Specify the license under which the project is released.)

Example:
This project is licensed under the MIT License - see the `LICENSE` file for details. (If you add a `LICENSE` file, ensure it's named appropriately, e.g., `LICENSE` or `LICENSE.md`).
---

*Note: The filenames `requirements.txt` and `app.py` are confirmed from the `ls` output. API key configuration details might need project-specific documentation.*