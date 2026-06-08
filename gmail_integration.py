"""
Gmail Integration for PhishGuard
Handles OAuth2 authentication and email fetching from the Gmail API.
"""

import os
import base64

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

BASE_DIR        = r'C:\Users\wongh\OneDrive\Documents\fyp2 code'
CREDENTIALS_FILE = os.path.join(BASE_DIR, 'credentials.json')
TOKEN_FILE       = os.path.join(BASE_DIR, 'token.json')


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

def is_credentials_file_present():
    return os.path.exists(CREDENTIALS_FILE)

def is_connected():
    return os.path.exists(TOKEN_FILE)

def get_gmail_service():
    """Return an authenticated Gmail API service. Opens browser on first run."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def disconnect():
    """Remove saved token so the user is logged out."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _decode_b64(data: str) -> str:
    """Decode a base64url-encoded Gmail body part."""
    data = data.replace('-', '+').replace('_', '/')
    pad = 4 - len(data) % 4
    if pad != 4:
        data += '=' * pad
    return base64.b64decode(data).decode('utf-8', errors='replace')

def _get_header(headers: list, name: str) -> str:
    for h in headers:
        if h['name'].lower() == name.lower():
            return h['value']
    return ''

def _extract_body(payload: dict) -> str:
    """
    Recursively walk a Gmail message payload and return the best body text.
    Prefers text/plain; falls back to text/html.
    """
    mime  = payload.get('mimeType', '')
    data  = payload.get('body', {}).get('data', '')
    parts = payload.get('parts', [])

    if mime == 'text/plain' and data:
        return _decode_b64(data)
    if mime == 'text/html' and data:
        return _decode_b64(data)

    plain, html = '', ''
    for part in parts:
        part_mime = part.get('mimeType', '')
        part_data = part.get('body', {}).get('data', '')
        if part_mime == 'text/plain' and part_data:
            plain += _decode_b64(part_data)
        elif part_mime == 'text/html' and part_data:
            html += _decode_b64(part_data)
        elif part.get('parts'):
            nested = _extract_body(part)
            if nested:
                plain += nested

    return plain or html


# ─────────────────────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_emails(service, max_results: int = 20, label: str = 'INBOX') -> list[dict]:
    """
    Fetch up to max_results emails from the given Gmail label.
    Returns a list of dicts compatible with PhishGuard's preprocess().
    """
    response = service.users().messages().list(
        userId='me',
        labelIds=[label],
        maxResults=max_results,
    ).execute()

    message_stubs = response.get('messages', [])
    emails = []

    for stub in message_stubs:
        msg = service.users().messages().get(
            userId='me',
            id=stub['id'],
            format='full',
        ).execute()

        payload = msg['payload']
        headers = payload.get('headers', [])

        emails.append({
            'id'      : stub['id'],
            'sender'  : _get_header(headers, 'From'),
            'receiver': _get_header(headers, 'To'),
            'subject' : _get_header(headers, 'Subject'),
            'date'    : _get_header(headers, 'Date'),
            'body'    : _extract_body(payload),
        })

    return emails
