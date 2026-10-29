# Sentinel URL Guard — AI Phishing URL Detector

## Included
- `app.py`: Streamlit app, training workflow, URL scanner, evaluation and dataset pages.
- `requirements.txt`: dependencies.
- No API key required.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. In the app, open **Train & Evaluate** and click **Train / Retrain Model**.
6. Open **URL Scanner** and enter a URL as text.

The public dataset is downloaded only when you choose that option. If unavailable, upload your own CSV with a URL column (`url`) and a label column (`label`).

## CSV labels
Supported strings include `good`/`bad`, `legitimate`/`phishing`, `benign`/`malicious`. Numeric labels use `0=legitimate` and `1=malicious`. Both classes are required.

Example:
```csv
url,label
https://example.com/,good
http://login-verify.example.invalid/account,bad
```

## Deploy to Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Go to https://share.streamlit.io/ and sign in.
3. Create an app, choose the repository/branch, and set main file path to `app.py`.
4. Deploy. No secrets or API key are needed.
5. Train the model from **Train & Evaluate**.

## How it works
Character-level TF-IDF features from URL text are passed to a Logistic Regression classifier. Evaluation uses a stratified hold-out split and reports accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.

## Important
This is an educational prototype, not production phishing protection. It does not visit the URL, inspect page content, check DNS/WHOIS, or consult live threat feeds. Scores are estimates, not calibrated probabilities. Never rely on it alone to decide if a site is safe. Do not enter credentials or sensitive data.

Dataset source: https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
