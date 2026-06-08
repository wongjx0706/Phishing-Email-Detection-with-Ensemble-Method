"""
PhishGuard — Phishing Email Detection Prototype
Steps 15 · 16 · 17
Run with:  streamlit run app.py
"""

import os, re, io, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from email import message_from_bytes, message_from_string
from scipy.sparse import hstack, csr_matrix

# ── Explainable AI (optional dependencies) ───────────────────────────────────
try:
    import shap
except Exception:
    shap = None
try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard — Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = r'C:\Users\wongh\OneDrive\Documents\fyp2 code\models'
OUTPUT_DIR = r'C:\Users\wongh\OneDrive\Documents\fyp2 code\output'

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD LISTS  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
URGENCY_WORDS = [
    'urgent','immediately','verify','update','confirm','suspend','limited',
    'expire','click here','act now','account','security','password','login',
    'bank','credit','prize','winner','congratulation','free','offer','risk',
    'warning','alert','suspended','unauthorized','validate','required','access',
]
SPAM_WORDS = [
    'viagra','cialis','pharmacy','prescription','pills','medication','casino',
    'lottery','million','billion','inheritance','nigeria','prince','transfer',
    'wire','bitcoin','crypto','investment','sex','adult','dating','weight loss',
    'cash','income','make money','work from home','get rich','fast cash',
    'no cost','click below','unsubscribe','dear friend','dear beneficiary',
]
FREE_EMAIL_PROVIDERS = {
    'gmail.com','yahoo.com','hotmail.com','outlook.com','aol.com',
    'protonmail.com','icloud.com','mail.com','ymail.com','live.com',
    'msn.com','inbox.com','zoho.com','gmx.com','fastmail.com',
    'rediffmail.com','yandex.com','mail.ru',
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def safe_str(val):
    return str(val) if val is not None else ''

def count_keywords(text, word_list):
    t = text.lower()
    return sum(1 for w in word_list if w in t)

def count_html_tags(text):
    return len(re.findall(r'<[^>]{1,300}>', text))

def count_anchor_tags(text):
    return len(re.findall(r'<a[\s>]', text, re.IGNORECASE))

def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\'{}|\\^`\[\]]+', text)

def has_ip_url(urls):
    pat = re.compile(r'https?://\d{1,3}(?:\.\d{1,3}){3}')
    return int(any(pat.match(u) for u in urls))

def has_suspicious_url(urls):
    markers = ['.tk','.ml','.ga','.cf','.gq','bit.ly','tinyurl','goo.gl','ow.ly']
    return int(any(m in u.lower() for u in urls for m in markers))

def extract_domain(email_str):
    m = re.search(r'@([\w.\-]+)', safe_str(email_str))
    return m.group(1).lower() if m else 'unknown'

def parse_date(date_str):
    fmts = [
        '%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z',
        '%d %b %Y %H:%M:%S %z',     '%a, %d %b %Y %H:%M:%S',
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(safe_str(date_str).strip(), fmt)
            return dt.hour, dt.weekday(), int(dt.weekday() >= 5)
        except (ValueError, AttributeError):
            continue
    return 12, 2, 0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15.1 — LOAD SAVED MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    def pkl(name):
        with open(os.path.join(MODEL_DIR, name), 'rb') as f:
            return pickle.load(f)
    rf       = pkl('model_rf.pkl')
    xgb      = pkl('model_xgb.pkl')
    lr       = pkl('model_lr.pkl')
    svm      = pkl('model_svm.pkl')
    pipeline = pkl('feature_pipeline.pkl')
    return rf, xgb, lr, svm, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15.3 — PREPROCESS INPUT
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_features(email_dict):
    """Extract engineered features from a parsed email dict."""
    body    = safe_str(email_dict.get('body', ''))
    subject = safe_str(email_dict.get('subject', ''))
    sender  = safe_str(email_dict.get('sender', ''))
    receiver= safe_str(email_dict.get('receiver', ''))
    date    = safe_str(email_dict.get('date', ''))
    text    = body + ' ' + subject
    urls    = extract_urls(body)
    url_lens= [len(u) for u in urls] if urls else [0]
    sender_domain   = extract_domain(sender)
    receiver_domain = extract_domain(receiver)
    hour, weekday, is_weekend = parse_date(date)

    feats = {
        # Textual
        'body_length'        : len(body),
        'subject_length'     : len(subject),
        'word_count'         : len(body.split()),
        'urgency_word_count' : count_keywords(text, URGENCY_WORDS),
        'spam_word_count'    : count_keywords(text, SPAM_WORDS),
        'exclamation_count'  : text.count('!'),
        'question_mark_count': text.count('?'),
        'capital_ratio'      : sum(c.isupper() for c in body) / max(len(body), 1),
        # URL / Structural
        'html_tag_count'     : count_html_tags(body),
        'has_html'           : int(bool(re.search(r'<html|<body|<div|<p[\s>]|<br', body, re.I))),
        'anchor_count'       : count_anchor_tags(body),
        'url_count_body'     : len(urls),
        'url_count_col'      : len(urls),
        'has_ip_url'         : has_ip_url(urls),
        'has_suspicious_url' : has_suspicious_url(urls),
        'max_url_length'     : max(url_lens),
        'special_char_count' : len(re.findall(r'[!@#$%^&*(){}\[\]|\\<>~`]', body)),
        # Header / Metadata
        'is_free_email'              : int(sender_domain in FREE_EMAIL_PROVIDERS),
        'sender_domain_length'       : len(sender_domain),
        'sender_has_numbers'         : int(bool(re.search(r'\d', sender_domain))),
        'hour_of_day'                : hour,
        'day_of_week'                : weekday,
        'is_weekend'                 : is_weekend,
        'sender_receiver_domain_match': int(
            sender_domain == receiver_domain and sender_domain != 'unknown'),
    }
    return feats

def preprocess(email_dict, pipeline):
    tfidf = pipeline['tfidf']
    text  = safe_str(email_dict.get('subject','')) + ' ' + safe_str(email_dict.get('body',''))
    X_tfidf = tfidf.transform([text])
    feats   = extract_all_features(email_dict)
    X_eng   = np.array([[feats[k] for k in pipeline['eng_feature_names']]],
                       dtype=np.float32)
    X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
    return X_combined.toarray().astype(np.float32), feats

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15.4 — RUN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(X_dense, rf, xgb, lr, svm):
    p_rf  = rf.predict_proba(X_dense)[0, 1]
    p_xgb = xgb.predict_proba(X_dense)[0, 1]
    p_lr  = lr.predict_proba(X_dense)[0, 1]
    p_svm = svm.predict_proba(X_dense)[0, 1]
    # Weighted soft voting: RF & XGB are the strongest models (~0.99 F1),
    # LR & SVM are weaker (~0.97), so give the stronger models more weight.
    w_rf, w_xgb, w_lr, w_svm = 2, 2, 1, 1
    avg = (w_rf*p_rf + w_xgb*p_xgb + w_lr*p_lr + w_svm*p_svm) / (w_rf + w_xgb + w_lr + w_svm)
    return avg, {'Random Forest': p_rf, 'XGBoost': p_xgb, 'Logistic Regression': p_lr, 'SVM': p_svm}

# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABLE AI  (Step 18) — SHAP for engineered features + LIME for words
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing SHAP explainer…")
def get_shap_explainer():
    """TreeExplainer on XGBoost (fast & exact for trees, ~0.99 F1 model)."""
    if shap is None:
        return None
    rf, xgb, lr, svm, pipeline = load_models()
    try:
        return shap.TreeExplainer(xgb)
    except Exception:
        return None

def shap_engineered_contributions(email_dict, pipeline):
    """Return [(feature_name, shap_value), ...] for the 24 engineered features."""
    explainer = get_shap_explainer()
    if explainer is None:
        return None
    X_dense, _ = preprocess(email_dict, pipeline)
    sv = explainer.shap_values(X_dense)
    if isinstance(sv, list):           # older SHAP: list per class
        sv = sv[-1]                    # positive (phishing) class
    arr = np.asarray(sv)
    if arr.ndim == 3:                  # (classes, n, features)
        arr = arr[-1]
    vals = arr[0]                      # (n_features,)
    eng_names = pipeline['eng_feature_names']
    n_eng = len(eng_names)
    eng_vals = vals[-n_eng:]           # engineered features are the LAST columns
    return list(zip(eng_names, [float(v) for v in eng_vals]))

def lime_word_contributions(email_dict, pipeline, rf, xgb, lr, svm,
                            n_features=12, num_samples=600):
    """Return [(word, weight), ...] — weight>0 pushes toward PHISHING."""
    if LimeTextExplainer is None:
        return None
    tfidf     = pipeline['tfidf']
    eng_names = pipeline['eng_feature_names']
    feats     = extract_all_features(email_dict)
    # Hold engineered features fixed so LIME isolates the effect of words only.
    X_eng_fixed = np.array([[feats[k] for k in eng_names]], dtype=np.float32)
    w_rf, w_xgb, w_lr, w_svm = 2, 2, 1, 1
    wsum = w_rf + w_xgb + w_lr + w_svm

    def classifier_fn(texts):
        Xt  = tfidf.transform(texts)
        eng = csr_matrix(np.repeat(X_eng_fixed, Xt.shape[0], axis=0))
        Xc  = hstack([Xt, eng]).toarray().astype(np.float32)
        p_rf  = rf.predict_proba(Xc)[:, 1]
        p_xgb = xgb.predict_proba(Xc)[:, 1]
        p_lr  = lr.predict_proba(Xc)[:, 1]
        p_svm = svm.predict_proba(Xc)[:, 1]
        avg = (w_rf*p_rf + w_xgb*p_xgb + w_lr*p_lr + w_svm*p_svm) / wsum
        return np.column_stack([1 - avg, avg])

    text = safe_str(email_dict.get('subject', '')) + ' ' + safe_str(email_dict.get('body', ''))
    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
    exp = explainer.explain_instance(text, classifier_fn,
                                     num_features=n_features, num_samples=num_samples,
                                     labels=(1,))   # always explain class 1 = Phishing
    # label=1 ensures weight>0 always means "pushed toward phishing" regardless of verdict.
    # Sort by |weight| descending so most impactful words appear first.
    words_sorted = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=True)
    return {
        'words': words_sorted,
        'proba': exp.predict_proba.tolist(),   # [P(Legitimate), P(Phishing)]
        'text' : text,
    }

def render_explanations(email_dict):
    """Render the 'Why this verdict?' panel (SHAP + LIME)."""
    rf, xgb, lr, svm, pipeline = load_models()
    with st.expander("🔍 Why this verdict? — Explainable AI (SHAP + LIME)", expanded=False):
        if shap is None and LimeTextExplainer is None:
            st.info("Install the XAI libraries to enable this:  `pip install shap lime`")
            return

        tab1, tab2 = st.tabs(["📊 SHAP — feature signals", "📝 LIME — suspicious words"])

        # ── SHAP: engineered red-flag features ───────────────────────────────
        with tab1:
            st.caption("How each engineered feature pushed THIS email's score. "
                       "Red = toward phishing, green = toward legitimate.")
            if shap is None:
                st.info("`shap` is not installed (`pip install shap`).")
            else:
                contribs = shap_engineered_contributions(email_dict, pipeline)
                if not contribs:
                    st.warning("SHAP explanation unavailable for this email.")
                else:
                    contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:12]
                    contribs = contribs[::-1]   # smallest at top for barh
                    names  = [c[0] for c in contribs]
                    vals   = [c[1] for c in contribs]
                    colors = ['#ff4b4b' if v > 0 else '#00b300' for v in vals]
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.barh(names, vals, color=colors)
                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.set_xlabel('SHAP value  (←legitimate   phishing→)')
                    ax.set_title('Top engineered-feature contributions (XGBoost)')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # ── LIME: words in the email text ────────────────────────────────────
        with tab2:
            if LimeTextExplainer is None:
                st.info("`lime` is not installed (`pip install lime`).")
            else:
                with st.spinner("Computing LIME word importances…"):
                    lime_result = lime_word_contributions(email_dict, pipeline, rf, xgb, lr, svm)
                if not lime_result:
                    st.warning("LIME explanation unavailable for this email.")
                else:
                    words  = lime_result['words']
                    proba  = lime_result['proba']   # [P(Legit), P(Phish)]
                    text   = lime_result['text']

                    col_prob, col_imp, col_text = st.columns([1, 1.2, 1.8])

                    # ── Left: prediction probabilities ────────────────────────
                    with col_prob:
                        st.markdown("**Prediction probabilities**")
                        fig, ax = plt.subplots(figsize=(2.5, 1.4))
                        labels_p = ['Legitimate', 'Phishing']
                        vals_p   = [proba[0], proba[1]]
                        bar_colors = ['#00b300', '#ff4b4b']
                        bars = ax.barh(labels_p, vals_p, color=bar_colors, height=0.45)
                        for bar, v in zip(bars, vals_p):
                            ax.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                                    f'{v:.2f}', va='center', fontsize=9)
                        ax.set_xlim(0, 1.15)
                        ax.set_xlabel('')
                        ax.spines[['top', 'right', 'left']].set_visible(False)
                        ax.tick_params(left=False)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    # ── Middle: word importance bar chart ─────────────────────
                    with col_imp:
                        st.markdown("**Word importances**")
                        top10 = words[:10][::-1]   # reverse so top word is at top of chart
                        wnames  = [w for w, _ in top10]
                        wvals   = [v for _, v in top10]
                        wcolors = ['#ff4b4b' if v > 0 else '#4b8bff' for v in wvals]
                        fig, ax = plt.subplots(figsize=(3, 3.5))
                        ax.barh(wnames, wvals, color=wcolors)
                        ax.axvline(0, color='black', linewidth=0.6)
                        ax.set_xlabel('Weight')
                        ax.spines[['top', 'right']].set_visible(False)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    # ── Right: email text with highlighted words ───────────────
                    with col_text:
                        st.markdown("**Text with highlighted words**")
                        word_color_map = {
                            w.lower(): ('#ffced0' if v > 0 else '#cdf0cd')
                            for w, v in words
                        }
                        def _highlight(match):
                            w = match.group(0)
                            color = word_color_map.get(w.lower())
                            if color:
                                return (f"<span style='background:{color};"
                                        f"border-radius:3px;padding:1px 3px'>{w}</span>")
                            return w
                        highlighted = re.sub(r'\b\w[\w\'-]*\b', _highlight, text)
                        st.markdown(
                            f"<div style='font-size:0.85rem;line-height:1.7;"
                            f"word-wrap:break-word;max-height:300px;overflow-y:auto'>"
                            f"{highlighted}</div>",
                            unsafe_allow_html=True
                        )

# ─────────────────────────────────────────────────────────────────────────────
# .EML PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_eml(raw_bytes):
    msg = message_from_bytes(raw_bytes)
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ('text/plain', 'text/html'):
                try:
                    body += part.get_payload(decode=True).decode(
                        part.get_content_charset() or 'utf-8', errors='replace')
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(
                msg.get_content_charset() or 'utf-8', errors='replace')
        except Exception:
            body = str(msg.get_payload())
    return {
        'sender'  : msg.get('From', ''),
        'receiver': msg.get('To', ''),
        'date'    : msg.get('Date', ''),
        'subject' : msg.get('Subject', ''),
        'body'    : body,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE EMAILS  (Step 16)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_EMAILS = {
    "🎣 Phishing — Nigerian Prince Scam": {
        'sender'  : 'Prince Adebayo <prince.adebayo@freemail.ng>',
        'receiver': 'victim@example.com',
        'date'    : 'Mon, 10 Mar 2025 09:14:22 +0000',
        'subject' : 'URGENT: Confidential Business Proposal — $15.5 Million USD',
        'body'    : (
            "Dear Friend,\n\nI am Prince Adebayo, son of late General Ibrahim Adebayo of Nigeria. "
            "I am contacting you with utmost confidence regarding a business transfer of USD $15,500,000.00 "
            "from our family account. Due to political instability, I urgently need a trusted foreign partner "
            "to receive this fund.\n\nYou will receive 30% of the total sum for your assistance. "
            "This is 100% risk free and legitimate. Please reply immediately with your bank account details "
            "to proceed with the transfer. Act now — time is limited!\n\n"
            "God Bless,\nPrince Adebayo\nLagos, Nigeria\n\n"
            "Click here to confirm: http://203.0.113.42/transfer/confirm?id=AB123"
        ),
    },
    "🎣 Phishing — Fake Bank Alert": {
        'sender'  : 'security-alert@paypa1-secure.tk',
        'receiver': 'user@gmail.com',
        'date'    : 'Wed, 15 Jan 2025 03:22:11 +0000',
        'subject' : 'Your account has been SUSPENDED — Verify immediately',
        'body'    : (
            "<html><body>"
            "<p>Dear Valued Customer,</p>"
            "<p>We have detected <b>unauthorized access</b> to your account. "
            "Your account has been <b>suspended</b> for security reasons.</p>"
            "<p>You must <b>verify your identity immediately</b> to restore access. "
            "Failure to verify within 24 hours will result in permanent account closure.</p>"
            "<a href='http://paypa1-secure.tk/verify?token=abc123'>Click here to verify now</a>"
            "<p>PayPal Security Team</p>"
            "</body></html>"
        ),
    },
    "🎣 Phishing — Lottery Winner": {
        'sender'  : 'lottery.winner2025@ymail.com',
        'receiver': 'user@outlook.com',
        'date'    : 'Fri, 07 Feb 2025 11:45:00 +0000',
        'subject' : 'CONGRATULATIONS! You have won £850,000 — Claim Now!',
        'body'    : (
            "CONGRATULATIONS!!!\n\n"
            "Your email address has been selected as the WINNER of the UK National Lottery "
            "Online Promo 2025. You have won the sum of £850,000.00 (Eight Hundred and Fifty Thousand Pounds).\n\n"
            "To claim your prize, send the following details immediately:\n"
            "- Full Name\n- Address\n- Phone Number\n- Bank Account Number\n\n"
            "Contact our claims agent: claim.agent@lottery-uk.ml\n"
            "Claim reference: UKL/2025/WINNER/0042\n\n"
            "WARNING: You must claim within 48 hours or prize will be forfeited!\n"
            "Get rich now! Fast cash guaranteed! No cost to you!\n\n"
            "http://bit.ly/claim-your-prize-now"
        ),
    },
    "✅ Legitimate — Work Meeting Invite": {
        'sender'  : 'alice.wong@company.com',
        'receiver': 'team@company.com',
        'date'    : 'Mon, 28 Apr 2025 08:30:00 +0800',
        'subject' : 'Team Sync — Tuesday 2pm',
        'body'    : (
            "Hi everyone,\n\n"
            "Just a reminder about our weekly team sync this Tuesday at 2:00 PM in Conference Room B.\n\n"
            "Agenda:\n"
            "1. Q2 project status update\n"
            "2. Resource allocation for May\n"
            "3. Any other business\n\n"
            "Please come prepared with your progress updates. "
            "Let me know if you have any agenda items to add.\n\n"
            "See you there!\nAlice\n\nAlice Wong | Project Manager | Company Inc.\nalice.wong@company.com"
        ),
    },
    "✅ Legitimate — GitHub Notification": {
        'sender'  : 'notifications@github.com',
        'receiver': 'developer@gmail.com',
        'date'    : 'Sun, 27 Apr 2025 22:10:45 +0000',
        'subject' : '[my-repo] Pull request merged: Fix null pointer exception in auth module',
        'body'    : (
            "Pull request #247 was merged into main by john-dev.\n\n"
            "Fix null pointer exception in auth module\n\n"
            "Changes:\n"
            "- Added null check before accessing user session\n"
            "- Added unit test for edge case\n"
            "- Updated documentation\n\n"
            "3 files changed, 42 insertions(+), 5 deletions(-)\n\n"
            "View it on GitHub: https://github.com/myorg/my-repo/pull/247\n\n"
            "You are receiving this because you are subscribed to this thread.\n"
            "Reply to this email directly, view it on GitHub, or unsubscribe."
        ),
    },
    "✅ Legitimate — University Newsletter": {
        'sender'  : 'newsletter@university.edu',
        'receiver': 'student@university.edu',
        'date'    : 'Thu, 24 Apr 2025 10:00:00 +0800',
        'subject' : 'Campus Update — April 2025',
        'body'    : (
            "Dear Students,\n\n"
            "Welcome to the April 2025 campus update.\n\n"
            "UPCOMING EVENTS:\n"
            "- Final Year Project Presentations: May 5-9, Main Hall\n"
            "- Career Fair 2025: May 15, Sports Complex\n"
            "- Library extended hours during exam period: 8am-midnight\n\n"
            "ACADEMIC REMINDERS:\n"
            "- Semester exam timetable is now available on the student portal\n"
            "- Assignment submission deadline: April 30\n\n"
            "For queries, contact the student affairs office at studentaffairs@university.edu\n\n"
            "Best regards,\nStudent Affairs Office"
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RESULT DISPLAY  (Step 15.5)
# ─────────────────────────────────────────────────────────────────────────────
def display_result(avg_proba, model_probas, feats, email_dict):
    is_phishing   = avg_proba >= 0.5
    confidence_pct = avg_proba * 100 if is_phishing else (1 - avg_proba) * 100

    # ── Main verdict banner ───────────────────────────────────────────────────
    if is_phishing:
        st.markdown(f"""
        <div style='background:#ff4b4b;padding:24px;border-radius:12px;text-align:center;margin-bottom:20px'>
            <h1 style='color:white;margin:0;font-size:2.2rem'>⚠️ PHISHING DETECTED</h1>
            <p style='color:#ffe0e0;font-size:1.2rem;margin:8px 0 0'>
                Confidence: <b>{confidence_pct:.1f}%</b> phishing probability
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#00b300;padding:24px;border-radius:12px;text-align:center;margin-bottom:20px'>
            <h1 style='color:white;margin:0;font-size:2.2rem'>✅ LEGITIMATE EMAIL</h1>
            <p style='color:#e0ffe0;font-size:1.2rem;margin:8px 0 0'>
                Confidence: <b>{confidence_pct:.1f}%</b> legitimate probability
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability gauge ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Ensemble Probability Score")
        fig, ax = plt.subplots(figsize=(5, 1.2))
        ax.barh([''], [avg_proba], color='#ff4b4b', height=0.5)
        ax.barh([''], [1 - avg_proba], left=[avg_proba], color='#00b300', height=0.5)
        ax.axvline(0.5, color='black', linewidth=1.5, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(f'Phishing probability: {avg_proba*100:.1f}%', fontsize=11)
        ax.text(0.02, 0, 'Legitimate', va='center', ha='left',
                fontsize=9, color='white', fontweight='bold')
        ax.text(0.98, 0, 'Phishing', va='center', ha='right',
                fontsize=9, color='white', fontweight='bold')
        ax.set_yticks([])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Per-Model Breakdown")
        model_df = pd.DataFrame({
            'Model'          : list(model_probas.keys()),
            'Phishing Prob'  : [f"{v*100:.1f}%" for v in model_probas.values()],
            'Vote'           : ['⚠️ Phishing' if v >= 0.5 else '✅ Legitimate'
                                for v in model_probas.values()],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    # ── Extracted email features ──────────────────────────────────────────────
    st.markdown("#### Extracted Email Features")
    risk_items = []
    if feats['urgency_word_count'] > 2: risk_items.append(f"🔴 High urgency language ({feats['urgency_word_count']} keywords)")
    if feats['spam_word_count'] > 1:    risk_items.append(f"🔴 Spam keywords found ({feats['spam_word_count']})")
    if feats['has_ip_url']:             risk_items.append("🔴 IP-based URL detected")
    if feats['has_suspicious_url']:     risk_items.append("🔴 Suspicious short/free URL detected")
    if feats['has_html']:               risk_items.append("🟡 HTML content present")
    if feats['is_free_email']:          risk_items.append("🟡 Sender uses free email provider")
    if feats['url_count_body'] > 3:     risk_items.append(f"🟡 Multiple URLs in body ({feats['url_count_body']})")
    if feats['capital_ratio'] > 0.3:    risk_items.append(f"🟡 Excessive capitalisation ({feats['capital_ratio']*100:.0f}%)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("URLs in body",      feats['url_count_body'])
        st.metric("Urgency keywords",  feats['urgency_word_count'])
        st.metric("Spam keywords",     feats['spam_word_count'])
    with c2:
        st.metric("HTML tags",         feats['html_tag_count'])
        st.metric("Exclamation marks", feats['exclamation_count'])
        st.metric("Word count",        feats['word_count'])
    with c3:
        st.metric("IP-based URL",      "Yes" if feats['has_ip_url'] else "No")
        st.metric("Free email sender", "Yes" if feats['is_free_email'] else "No")
        st.metric("Capital ratio",     f"{feats['capital_ratio']*100:.1f}%")

    if risk_items:
        st.markdown("**Risk Indicators Found:**")
        for item in risk_items:
            st.markdown(f"- {item}")
    else:
        st.success("No significant risk indicators found.")

    # ── Explainable AI: why this verdict? ─────────────────────────────────────
    render_explanations(email_dict)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛡️ PhishGuard")
    st.markdown("**Phishing Email Detection System**")
    st.markdown("---")
    st.markdown("### Model Architecture")
    st.markdown("""
    **Base Models:**
    - 🌲 Random Forest
    - ⚡ XGBoost
    - 📐 Logistic Regression
    - ✂️ SVM (LinearSVC + Calibration)

    **Ensemble:** Weighted Soft Voting
    (RF×2, XGB×2, LR×1, SVM×1)

    **Features:** TF-IDF (2000 bigrams)
    + 24 engineered features
    """)
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    - CEAS 2008 spam corpus
    - Nigerian scam emails
    - Nazario phishing corpus
    - **~165,000 emails total**
    """)
    st.markdown("---")
    try:
        rf, xgb, lr, svm, pipeline = load_models()
        st.success("✅ Models loaded")
    except Exception as e:
        st.error(f"❌ Models not found\n{e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect Email", "🧪 Test Samples", "📊 Model Results", "📬 Gmail Inbox"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT EMAIL  (Steps 15.2 – 15.5)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Email Phishing Detection")
    st.markdown("Paste raw email content or upload a `.eml` file to check if it is phishing.")

    input_method = st.radio("Input method", ["✏️ Paste email text", "📁 Upload .eml file"],
                             horizontal=True)
    email_dict = None

    # ── Step 15.2 — Build Input Interface ────────────────────────────────────
    if input_method == "✏️ Paste email text":
        col_l, col_r = st.columns(2)
        with col_l:
            sender   = st.text_input("From (sender)",   placeholder="sender@example.com")
            receiver = st.text_input("To (receiver)",   placeholder="receiver@example.com")
        with col_r:
            subject  = st.text_input("Subject",         placeholder="Email subject line")
            date_str = st.text_input("Date (optional)", placeholder="Mon, 28 Apr 2025 09:00:00 +0800")
        body = st.text_area("Email Body", height=250,
                             placeholder="Paste the full email body here…")
        if st.button("🔍 Analyse Email", type="primary", use_container_width=True):
            if not body.strip():
                st.warning("Please enter an email body before analysing.")
            else:
                email_dict = {
                    'sender': sender, 'receiver': receiver,
                    'subject': subject, 'date': date_str, 'body': body,
                }

    else:  # .eml upload
        uploaded = st.file_uploader("Upload .eml file", type=['eml'])
        if uploaded is not None:
            email_dict = parse_eml(uploaded.read())
            st.success("File parsed successfully.")
            with st.expander("Preview parsed email"):
                st.write(f"**From:** {email_dict['sender']}")
                st.write(f"**To:** {email_dict['receiver']}")
                st.write(f"**Subject:** {email_dict['subject']}")
                st.write(f"**Date:** {email_dict['date']}")
                st.text_area("Body preview", email_dict['body'][:1000], height=200, disabled=True)
            if st.button("🔍 Analyse Email", type="primary", use_container_width=True):
                pass  # email_dict already set

    # ── Steps 15.3–15.5 — Preprocess → Predict → Display ────────────────────
    if email_dict and email_dict.get('body', '').strip():
        with st.spinner("Analysing email…"):
            try:
                X_dense, feats = preprocess(email_dict, pipeline)
                avg_proba, model_probas = predict(X_dense, rf, xgb, lr, svm)
                st.markdown("---")
                display_result(avg_proba, model_probas, feats, email_dict)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROTOTYPE TESTING  (Step 16)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Step 16 — Prototype Testing")
    st.markdown("Select a pre-loaded sample email to verify end-to-end detection.")

    sample_name = st.selectbox("Choose a sample email", list(SAMPLE_EMAILS.keys()))
    sample      = SAMPLE_EMAILS[sample_name]

    with st.expander("📧 View sample email content", expanded=True):
        st.write(f"**From:** {sample['sender']}")
        st.write(f"**To:** {sample['receiver']}")
        st.write(f"**Subject:** {sample['subject']}")
        body_preview = sample['body'][:800] + ('…' if len(sample['body']) > 800 else '')
        st.text_area("Body", body_preview, height=180, disabled=True)

    if st.button("🔍 Run Detection on Sample", type="primary", use_container_width=True):
        with st.spinner("Analysing…"):
            X_dense, feats = preprocess(sample, pipeline)
            avg_proba, model_probas = predict(X_dense, rf, xgb, lr, svm)
        st.markdown("---")
        display_result(avg_proba, model_probas, feats, sample)

    # Batch test all samples
    st.markdown("---")
    st.markdown("### Batch Test — All Sample Emails")
    if st.button("▶ Run All Samples", use_container_width=True):
        results = []
        for name, email in SAMPLE_EMAILS.items():
            X_dense, feats = preprocess(email, pipeline)
            avg_proba, model_probas = predict(X_dense, rf, xgb, lr, svm)
            expected = "Phishing" if name.startswith("🎣") else "Legitimate"
            predicted = "Phishing" if avg_proba >= 0.5 else "Legitimate"
            results.append({
                'Sample'          : name,
                'Expected'        : expected,
                'Predicted'       : predicted,
                'Confidence'      : f"{max(avg_proba, 1-avg_proba)*100:.1f}%",
                'Phishing Prob'   : f"{avg_proba*100:.1f}%",
                'Correct'         : "✅" if expected == predicted else "❌",
            })
        df = pd.DataFrame(results)
        correct = sum(1 for r in results if r['Correct'] == "✅")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.metric("Prototype Accuracy on Samples",
                  f"{correct}/{len(results)} correct ({correct/len(results)*100:.0f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS & DOCUMENTATION  (Step 17)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Step 17 — Results & Documentation")
    st.markdown("All training results, charts, and metrics for the FYP report.")

    # ── Metrics table ─────────────────────────────────────────────────────────
    metrics_csv = os.path.join(OUTPUT_DIR, 'metrics_comparison.csv')
    if os.path.exists(metrics_csv):
        st.markdown("### Model Performance Comparison")
        df_metrics = pd.read_csv(metrics_csv, index_col=0)
        st.dataframe(
            df_metrics.style.format('{:.4f}')
                            .highlight_max(color='#c6efce')
                            .highlight_min(color='#ffc7ce'),
            use_container_width=True,
        )
        csv_bytes = df_metrics.to_csv().encode()
        st.download_button("⬇ Download metrics CSV", csv_bytes,
                           "metrics_comparison.csv", "text/csv")
    else:
        st.info("Run the training pipeline first to generate metrics.")

    # ── Dataset Preview (Step 2 EDA) ─────────────────────────────────────────
    st.markdown("### Dataset Preview (EDA)")
    DATASET_PATH = r'C:\Users\wongh\OneDrive\Documents\fyp2 code\COMBINED_ceas08_nigerian_nazario.csv'
    if os.path.exists(DATASET_PATH):
        @st.cache_data(show_spinner=False)
        def load_dataset_preview():
            df = pd.read_csv(DATASET_PATH)
            return df

        df_full = load_dataset_preview()
        total   = len(df_full)
        n_phish = int((df_full['label'] == 1).sum())
        n_legit = int((df_full['label'] == 0).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total emails", f"{total:,}")
        c2.metric("Phishing (label=1)", f"{n_phish:,}", f"{n_phish/total*100:.1f}%")
        c3.metric("Legitimate (label=0)", f"{n_legit:,}", f"{n_legit/total*100:.1f}%")

        with st.expander("📋 Sample rows — first 10 emails", expanded=False):
            preview_df = df_full.head(10).copy()
            for col in ('body', 'subject'):
                if col in preview_df.columns:
                    preview_df[col] = preview_df[col].astype(str).str[:120] + '…'
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

        with st.expander("📋 Sample rows — 5 phishing + 5 legitimate", expanded=False):
            phish_sample = df_full[df_full['label'] == 1].head(5)
            legit_sample  = df_full[df_full['label'] == 0].head(5)
            mixed = pd.concat([phish_sample, legit_sample]).reset_index(drop=True).copy()
            mixed['label'] = mixed['label'].map({1: '⚠️ Phishing', 0: '✅ Legitimate'})
            for col in ('body', 'subject'):
                if col in mixed.columns:
                    mixed[col] = mixed[col].astype(str).str[:120] + '…'
            st.dataframe(mixed, use_container_width=True, hide_index=True)

        with st.expander("📊 Column statistics", expanded=False):
            stats = pd.DataFrame({
                'Column'     : df_full.columns.tolist(),
                'Non-null'   : df_full.notna().sum().values,
                'Null'       : df_full.isna().sum().values,
                'Dtype'      : df_full.dtypes.astype(str).values,
                'Sample value': [str(df_full[c].iloc[0])[:80] for c in df_full.columns],
            })
            st.dataframe(stats, use_container_width=True, hide_index=True)
    else:
        st.info(f"Dataset not found at `{DATASET_PATH}`.")

    st.markdown("---")

    # ── Saved plots ────────────────────────────────────────────────────────────
    plot_files = {
        "Class Distribution & Body Length (EDA)"   : '01_eda.png',
        "Class Balance Before/After SMOTE"          : '02_smote_balance.png',
        "Confusion Matrices — Base Models"          : '03_confusion_matrices_base.png',
        "Ensemble Confusion Matrix & ROC Curves"    : '04_ensemble_roc.png',
        "Model Performance Comparison"              : '05_model_comparison.png',
        "Random Forest Feature Importance"          : '06_rf_feature_importance.png',
    }

    st.markdown("### Plots & Charts")
    for title, fname in plot_files.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            with st.expander(f"📊 {title}"):
                st.image(fpath, use_column_width=True)
                with open(fpath, 'rb') as f:
                    st.download_button(f"⬇ Download {fname}", f.read(),
                                       fname, "image/png", key=fname)
        else:
            st.caption(f"⏳ {title} — not generated yet (run training pipeline first)")

    # ── Findings summary ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Key Findings Summary")
    if os.path.exists(metrics_csv):
        df_m = pd.read_csv(metrics_csv, index_col=0)
        best_model = df_m['F1-Score'].idxmax()
        best_f1    = df_m['F1-Score'].max()
        best_auc   = df_m.loc['Soft Voting Ensemble', 'ROC-AUC'] if 'Soft Voting Ensemble' in df_m.index else 0
        st.markdown(f"""
        | Finding | Value |
        |---|---|
        | Best individual model | **{best_model}** (F1 = {best_f1:.4f}) |
        | Ensemble ROC-AUC | **{best_auc:.4f}** |
        | Training strategy | SMOTE + Stratified 80/20 split |
        | Feature approach | Hybrid (TF-IDF bigrams + 24 engineered) |
        | Ensemble method | Soft Voting (average of 4 model probabilities) |
        | Dataset size | 44,051 emails (CEAS08 + Nigerian + Nazario) |
        """)
    else:
        st.markdown("""
        | Finding | Value |
        |---|---|
        | Training strategy | SMOTE + Stratified 80/20 split |
        | Feature approach | Hybrid (TF-IDF bigrams + 24 engineered) |
        | Ensemble method | Soft Voting (4 base models) |
        | Dataset size | 44,051 emails |
        """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — GMAIL INBOX SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Gmail Inbox Scanner")
    st.markdown("Connect your Gmail account and scan real emails for phishing threats.")

    # ── Try importing the Gmail library ──────────────────────────────────────
    try:
        import gmail_integration as gm
        gmail_available = True
    except ImportError:
        gmail_available = False

    if not gmail_available:
        st.error("Gmail libraries are not installed.")
        st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib", language="bash")
        st.stop()

    # ── Step 1: credentials.json check ───────────────────────────────────────
    if not gm.is_credentials_file_present():
        st.warning("**credentials.json not found.** Follow the setup steps below to connect Gmail.")
        with st.expander("📋 How to set up Gmail API (click to expand)", expanded=True):
            st.markdown("""
**Step 1 — Create a Google Cloud project**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project**
3. Name it `PhishGuard` and click **Create**

**Step 2 — Enable the Gmail API**
1. In your project, go to **APIs & Services → Library**
2. Search for **Gmail API** and click **Enable**

**Step 3 — Create OAuth credentials**
1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. If prompted, configure the **OAuth consent screen** first:
   - User Type: **External**
   - App name: `PhishGuard`
   - Add your own Gmail address as a **Test user**
4. Back on Create Credentials:
   - Application type: **Desktop app**
   - Name: `PhishGuard Desktop`
   - Click **Create**
5. Download the JSON file → click **Download JSON**

**Step 4 — Place the file**
- Rename the downloaded file to **`credentials.json`**
- Put it here:
""")
            st.code(r"C:\Users\wongh\OneDrive\Documents\fyp2 code\credentials.json")
            st.info("Refresh this page after placing the file.")
        st.stop()

    # ── Step 2: Connection status + connect/disconnect ────────────────────────
    st.markdown("---")
    col_status, col_btn = st.columns([3, 1])

    with col_status:
        if gm.is_connected():
            st.success("Gmail connected. Your token is saved locally.")
        else:
            st.info("Not connected. Click **Connect Gmail** to log in.")

    with col_btn:
        if gm.is_connected():
            if st.button("Disconnect", use_container_width=True):
                gm.disconnect()
                st.session_state.pop('gmail_emails', None)
                st.rerun()
        else:
            if st.button("Connect Gmail", type="primary", use_container_width=True):
                with st.spinner("Opening browser for Google login…"):
                    try:
                        gm.get_gmail_service()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")

    if not gm.is_connected():
        st.stop()

    # ── Step 3: Fetch options ─────────────────────────────────────────────────
    st.markdown("---")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        num_emails = st.slider("Number of emails to fetch", 5, 50, 20, step=5)
    with col_opt2:
        label_choice = st.selectbox("Mailbox label", ["INBOX", "SPAM", "UNREAD"])
    with col_opt3:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_clicked = st.button("Fetch & Scan Emails", type="primary", use_container_width=True)

    # ── Step 4: Fetch + classify ──────────────────────────────────────────────
    if fetch_clicked:
        with st.spinner(f"Fetching {num_emails} emails from {label_choice}…"):
            try:
                service = gm.get_gmail_service()
                emails  = gm.fetch_emails(service, max_results=num_emails, label=label_choice)
                st.session_state['gmail_emails'] = emails
            except Exception as e:
                st.error(f"Failed to fetch emails: {e}")
                st.exception(e)

    emails = st.session_state.get('gmail_emails', [])

    if not emails:
        st.caption("No emails fetched yet. Press **Fetch & Scan Emails** above.")
        st.stop()

    # ── Step 5: Classify all fetched emails ───────────────────────────────────
    st.markdown(f"### Scan Results — {len(emails)} emails")

    rows = []
    for email in emails:
        try:
            X_dense, feats = preprocess(email, pipeline)
            avg_proba, _   = predict(X_dense, rf, xgb, lr, svm)
        except Exception:
            avg_proba = -1.0  # mark as error

        verdict = (
            "⚠️ Phishing"  if avg_proba >= 0.5
            else "✅ Safe"  if avg_proba >= 0
            else "❓ Error"
        )
        raw_body = email.get('body', '')
        body_preview = ' '.join(raw_body.split())[:200]
        if len(raw_body.split()) > 40:
            body_preview += '…'

        rows.append({
            'Verdict'      : verdict,
            'Phishing %'   : f"{avg_proba*100:.1f}%" if avg_proba >= 0 else "—",
            'From'         : email.get('sender',  '')[:60],
            'Subject'      : email.get('subject', '')[:70],
            'Body Preview' : body_preview,
            'Date'         : email.get('date',    '')[:30],
            '_idx'         : len(rows),
            '_proba'       : avg_proba,
        })

    df_results = pd.DataFrame(rows)

    phishing_count = sum(1 for r in rows if r['_proba'] >= 0.5)
    safe_count     = sum(1 for r in rows if 0 <= r['_proba'] < 0.5)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total scanned",    len(rows))
    m2.metric("Phishing detected", phishing_count, delta=None)
    m3.metric("Safe",             safe_count)

    display_df = df_results.drop(columns=['_idx', '_proba'])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Step 6: Deep-dive into a single email ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Deep Analysis — Pick an Email")

    subjects = [f"{r['Verdict']}  |  {r['Subject'] or '(no subject)'}  |  {r['From'][:40]}"
                for r in rows]
    selected_idx = st.selectbox("Select email to analyse", range(len(subjects)),
                                format_func=lambda i: subjects[i])

    if st.button("Run Full Analysis", type="primary", use_container_width=True):
        chosen_email = emails[selected_idx]

        with st.expander("Email preview", expanded=False):
            st.write(f"**From:** {chosen_email.get('sender','')}")
            st.write(f"**To:** {chosen_email.get('receiver','')}")
            st.write(f"**Subject:** {chosen_email.get('subject','')}")
            st.write(f"**Date:** {chosen_email.get('date','')}")
            st.text_area("Body (first 1000 chars)",
                         chosen_email.get('body','')[:1000], height=200, disabled=True)

        with st.spinner("Analysing…"):
            X_dense, feats = preprocess(chosen_email, pipeline)
            avg_proba, model_probas = predict(X_dense, rf, xgb, lr, svm)

        st.markdown("---")
        display_result(avg_proba, model_probas, feats, chosen_email)
