"""
src/utils/mappls_auth.py
Mappls OAuth2 token management. NOT imported by main pipeline.
Only called explicitly: python -m src.data.mappls_traffic --city gnr
Credentials read from env at call time — safe to import without credentials set.
"""
import os, time, logging, requests
logger = logging.getLogger(__name__)
_TOKEN_URL = "https://outpost.mappls.com/api/security/oauth/token"
_REFRESH_MARGIN_SECONDS = 120
_token_cache = None

class AuthError(Exception):
    pass

def _fetch_token(client_id, client_secret):
    try:
        resp = requests.post(_TOKEN_URL, data={"grant_type":"client_credentials","client_id":client_id,"client_secret":client_secret}, timeout=15)
    except requests.RequestException as exc:
        raise AuthError(f"Network error: {exc}") from exc
    if resp.status_code != 200:
        raise AuthError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    try: data = resp.json()
    except ValueError as exc:
        raise AuthError(f"Invalid JSON: {resp.text[:200]}") from exc
    for k in ("access_token","expires_in"):
        if k not in data: raise AuthError(f"Missing '{k}': {data}")
    return data

def _is_token_fresh(cache):
    return time.time() < cache["expires_at"] - _REFRESH_MARGIN_SECONDS

def get_bearer_token():
    global _token_cache
    client_id = os.environ.get("MAPPLS_CLIENT_ID","").strip()
    client_secret = os.environ.get("MAPPLS_CLIENT_SECRET","").strip()
    if not client_id or not client_secret:
        raise AuthError("MAPPLS_CLIENT_ID and MAPPLS_CLIENT_SECRET must be set. Register at https://auth.mappls.com/console")
    if _token_cache is not None and _is_token_fresh(_token_cache):
        return _token_cache["access_token"]
    data = _fetch_token(client_id, client_secret)
    issued_at = time.time()
    _token_cache = {"access_token": data["access_token"], "expires_at": issued_at + int(data["expires_in"])}
    return _token_cache["access_token"]

def get_rest_api_key():
    key = os.environ.get("MAPPLS_REST_API_KEY","").strip()
    if not key:
        raise AuthError("MAPPLS_REST_API_KEY must be set. Obtain from https://auth.mappls.com/console")
    return key

def get_auth_headers():
    return {"Authorization": f"Bearer {get_bearer_token()}"}

def invalidate_token_cache():
    global _token_cache
    _token_cache = None
