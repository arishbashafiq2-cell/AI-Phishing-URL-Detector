import re
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Sentinel URL Guard", page_icon="🛡️", layout="wide")
st.title("🛡️ Sentinel URL Guard")
st.subheader("AI Phishing URL Detector")
st.warning("Educational prototype only. This app analyzes URL text and does not open or visit the URL. A low-risk result is not proof that a website is safe.")

DATA_URL = "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv"
MAX_ROWS = 30000
if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "data" not in st.session_state:
    st.session_state.data = None

def label_value(v):
    s = str(v).strip().lower()
    if s in {"bad", "phishing", "malicious", "malware", "defacement", "1", "true"}:
        return 1
    if s in {"good", "legitimate", "benign", "safe", "clean", "0", "false"}:
        return 0
    try:
        n = int(float(s))
        return n if n in (0, 1) else None
    except Exception:
        return None

def prepare(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    uc = next((cols[k] for k in ("url", "urls", "website", "link") if k in cols), None)
    lc = next((cols[k] for k in ("label", "type", "class", "status", "target") if k in cols), None)
    if uc is None or lc is None:
        raise ValueError("CSV must contain a URL column (e.g. url) and a label column (e.g. label).")
    out = df[[uc, lc]].copy()
    out.columns = ["url", "raw_label"]
    out["url"] = out["url"].astype(str).str.strip()
    out["label"] = out["raw_label"].map(label_value)
    out = out.dropna(subset=["label"])
    out = out[out["url"].str.len().between(4, 2048)]
    out = out[out["url"].str.lower() != "nan"].drop_duplicates("url")
    out["label"] = out["label"].astype(int)
    out = out[["url", "label"]]
    if out["label"].nunique() != 2:
        raise ValueError("Dataset must contain both classes. Use good/bad, legitimate/phishing, or 0/1 labels (1=malicious).")
    return out

def fit_model(data):
    if len(data) > MAX_ROWS:
        data, _ = train_test_split(data, train_size=MAX_ROWS, random_state=42, stratify=data["label"])
    xtr, xte, ytr, yte = train_test_split(data["url"], data["label"], test_size=.2, random_state=42, stratify=data["label"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42))
    ])
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    prob = model.predict_proba(xte)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1-score": f1_score(yte, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(yte, prob),
        "confusion": confusion_matrix(yte, pred, labels=[0, 1]),
        "report": classification_report(yte, pred, labels=[0, 1], target_names=["Legitimate", "Potentially malicious"], output_dict=True, zero_division=0),
        "train_n": len(xtr), "test_n": len(xte)
    }
    return model, metrics, data

def normalize_url(text):
    text = text.strip()
    if not text:
        raise ValueError("Enter a URL first.")
    if any(ch.isspace() for ch in text):
        raise ValueError("URL must not contain spaces.")
    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid URL, for example https://example.com")
    return text

def signals(url):
    p = urlparse(url)
    host = p.netloc.split("@")[-1].split(":")[0].lower()
    items = []
    if len(url) > 75: items.append(("Long URL", f"The URL has {len(url)} characters."))
    if "@" in url: items.append(("At symbol", "May obscure the intended destination."))
    if re.search(r"https?://\d{1,3}(?:\.\d{1,3}){3}", url): items.append(("IP-address host", "Uses an IP address rather than a typical domain name."))
    if "-" in host: items.append(("Hyphen in hostname", "Can be normal, but inspect the domain carefully."))
    if host.count(".") >= 4: items.append(("Many subdomain levels", "Several hostname levels can disguise the actual domain."))
    if any(t in url.lower() for t in ("login", "verify", "account", "password", "signin", "secure", "update")):
        items.append(("Account-related wording", "Common in legitimate pages too, but also seen in phishing URLs."))
    if not url.lower().startswith("https://"): items.append(("No HTTPS prefix", "This alone does not prove a URL is malicious."))
    if not items: items.append(("No obvious lexical warning", "This does not prove the URL is safe."))
    return host, items

with st.sidebar:
    page = st.radio("Navigation", ["URL Scanner", "Train & Evaluate", "Dataset", "About"])
    st.divider()
    st.caption("Model status")
    st.success("Trained") if st.session_state.model is not None else st.info("Not trained yet")
    st.caption("No API key required")

if page == "URL Scanner":
    st.markdown("Paste a URL as text. The app will not visit it.")
    with st.form("scan"):
        url_input = st.text_input("Website URL", placeholder="https://example.com/login")
        submitted = st.form_submit_button("🔍 Analyze URL", type="primary")
    if submitted:
        try:
            url = normalize_url(url_input)
            host, items = signals(url)
            a, b = st.columns(2)
            a.metric("Hostname", host or "Unknown")
            b.metric("URL length", f"{len(url)} characters")
            if st.session_state.model is None:
                st.info("Train the model first in Train & Evaluate. Basic URL signals are shown below.")
            else:
                score = float(st.session_state.model.predict_proba([url])[0][1])
                if score >= .75:
                    verdict, emoji = "High risk", "🔴"
                elif score >= .45:
                    verdict, emoji = "Elevated risk", "🟠"
                else:
                    verdict, emoji = "Lower model-estimated risk", "🟢"
                st.markdown(f"## {emoji} {verdict}")
                st.metric("Model-estimated malicious score", f"{score:.1%}")
                st.progress(min(max(score, 0), 1), text="Estimated malicious-class score")
                st.caption("This is not a calibrated real-world probability. Model predictions can be wrong.")
            st.markdown("### URL pattern review")
            for title, detail in items:
                st.info(f"**{title}:** {detail}")
        except ValueError as e:
            st.error(str(e))
    st.caption("Never enter passwords or sensitive information. Do not rely on this tool alone to decide if a website is safe.")

elif page == "Train & Evaluate":
    st.write("Train a character-level TF-IDF + Logistic Regression model using labeled URL examples.")
    source = st.radio("Dataset source", ["Download public sample dataset", "Upload my own CSV"], horizontal=True)
    upload = st.file_uploader("Upload CSV (URL and label columns required)", type=["csv"]) if source == "Upload my own CSV" else None
    st.caption(f"At most {MAX_ROWS:,} URLs are sampled to keep training manageable.")
    if st.button("🚀 Train / Retrain Model", type="primary"):
        try:
            with st.spinner("Loading and cleaning labeled data..."):
                if source == "Download public sample dataset":
                    raw = pd.read_csv(DATA_URL, low_memory=False)
                    name = "public GitHub dataset"
                else:
                    if upload is None:
                        st.error("Upload a CSV first.")
                        st.stop()
                    raw = pd.read_csv(upload)
                    name = upload.name
                data = prepare(raw)
            with st.spinner("Training and evaluating..."):
                model, metrics, used = fit_model(data)
            st.session_state.model, st.session_state.metrics, st.session_state.data = model, metrics, used
            st.session_state.dataset_name = name
            st.success(f"Training complete using {len(used):,} labeled URLs.")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.info("If the public dataset is unavailable, upload your own CSV with URL and label columns.")
    m = st.session_state.metrics
    if m:
        st.markdown("### Held-out test performance")
        cols = st.columns(5)
        for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
            col.metric(key, f"{m[key]:.3f}")
        st.caption(f"Training rows: {m['train_n']:,} · Test rows: {m['test_n']:,}. Metrics reflect this dataset and split only.")
        cm = pd.DataFrame(m["confusion"], index=["Actual legitimate", "Actual malicious"], columns=["Predicted legitimate", "Predicted malicious"])
        st.markdown("#### Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto", title="Confusion matrix"), use_container_width=True)
        st.markdown("#### Classification report")
        st.dataframe(pd.DataFrame(m["report"]).transpose().round(3), use_container_width=True)

elif page == "Dataset":
    data = st.session_state.data
    if data is None:
        st.info("Train a model first to load and preview a dataset.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("URLs used", f"{len(data):,}")
        c2.metric("Unique URLs", f"{data['url'].nunique():,}")
        counts = data["label"].map({0: "Legitimate", 1: "Potentially malicious"}).value_counts().rename_axis("Class").reset_index(name="Count")
        st.plotly_chart(px.bar(counts, x="Class", y="Count", title="Label distribution"), use_container_width=True)
        shown = data.copy()
        shown["label"] = shown["label"].map({0: "Legitimate", 1: "Potentially malicious"})
        st.dataframe(shown.head(100), use_container_width=True)
        st.download_button("Download sampled dataset", data.to_csv(index=False).encode(), "sampled_url_dataset.csv", "text/csv")

else:
    st.markdown("""
    ### Project overview
    **Goal:** Demonstrate supervised machine learning for URL-based phishing screening.

    **Workflow**
    1. Load a labeled URL dataset.
    2. Normalize labels to `0 = legitimate`, `1 = potentially malicious`.
    3. Convert character n-grams into TF-IDF features.
    4. Train Logistic Regression and evaluate on a held-out test set.
    5. Score a user-entered URL and show simple lexical signals.

    **Stack:** Python, Streamlit, Pandas, Scikit-learn, Plotly.

    **Limitations:** The app only analyzes URL text. It does not fetch page content, check DNS/WHOIS, consult live threat feeds, or guarantee safety. Dataset bias and stale labels can affect performance. Do not use it as a production security control.
    """)
    st.markdown("Dataset source: [faizann24 malicious URL dataset on GitHub](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs)")
