# -*- coding: utf-8 -*-  
"""  
Stateless RAG API (Flask) - Entra ID auth for Azure resources  
  
- Incoming auth: Azure App Service EasyAuth (X-MS-CLIENT-PRINCIPAL)  
  - local: bypass allowed ONLY from localhost remote_addr (APP_ENV=local)  
  - prod: can enforce EasyAuth enabled via REQUIRE_EASYAUTH=true  
  
- Azure OpenAI: Entra ID (azure_ad_token_provider)  
- Azure AI Search: Entra ID (TokenCredential)  
- Azure Blob: Entra ID + User Delegation SAS (download_url) [optional]  
  
Main endpoint:  
  POST /v1/rag  
  {  
    "query": "...",  
    "history": [{"role":"user|assistant","content":"..."}],  
    "model": "...",                 # optional (deployment name recommended)  
    "system_prompt": "...",         # optional  
    "reasoning_effort": "low|medium|high",  # optional  
    "search": {  
      "index": "...",  
      "container": "...",  
      "top_k": 15,  
      "rewrite_queries": true,  
      "use_vector": true,           # optional (default: DEFAULT_USE_VECTOR_SEARCH)  
      "include_sas": false          # optional (default: DEFAULT_INCLUDE_SAS)  
    }  
  }  
  
Search behavior (default):  
- multi-query  
- per query: keyword(simple) + semantic(rerank) + (optional) vector  
- RRF fusion  
- chunk de-duplication by chunk_id (or filepath::chunk_id)  
"""  
  
from __future__ import annotations  
  
import base64  
import json  
import logging  
import os  
import threading  
import time  
import uuid  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from dataclasses import dataclass  
from datetime import datetime, timedelta, timezone  
from typing import Any, Optional  
from urllib.parse import quote  
  
from flask import Flask, Response, jsonify, request  
from werkzeug.exceptions import BadRequest  
  
from azure.identity import AzureCliCredential, ManagedIdentityCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizedQuery  
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas  
from openai import AzureOpenAI  
  
# -------------------------  
# Logging  
# -------------------------  
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))  
logger = logging.getLogger("rag_api")  
  
# -------------------------  
# Config helpers  
# -------------------------  
def _env(name: str, default: Optional[str] = None) -> str:  
    v = os.getenv(name, default)  
    return v if v is not None else ""  
  
  
def _env_int(name: str, default: int) -> int:  
    try:  
        return int(_env(name, str(default)))  
    except Exception:  
        return default  
  
  
def _env_bool(name: str, default: bool) -> bool:  
    v = (_env(name, "true" if default else "false") or "").strip().lower()  
    return v in ("1", "true", "yes", "y", "on")  
  
  
def _csv_env(name: str, default_csv: str) -> set[str]:  
    raw = _env(name, default_csv)  
    return {x.strip() for x in raw.split(",") if x.strip()}  
  
  
def _bool(v: Any, default: bool) -> bool:  
    if v is None:  
        return default  
    if isinstance(v, bool):  
        return v  
    if isinstance(v, str):  
        return v.strip().lower() in ("1", "true", "yes", "y", "on")  
    return bool(v)  
  
  
APP_ENV = (_env("APP_ENV", "prod") or "prod").lower()  
IS_LOCAL = APP_ENV == "local"  
  
# EasyAuth enforcement  
REQUIRE_EASYAUTH = _env_bool("REQUIRE_EASYAUTH", default=(not IS_LOCAL))  
LOCALHOST_REMOTE_ADDRS = {"127.0.0.1", "::1"}  
  
# Defaults / allowlists  
DEFAULT_MODEL = _env("DEFAULT_MODEL", "gpt-5.1")  
ALLOWED_MODELS = _csv_env("ALLOWED_MODELS", DEFAULT_MODEL) or {DEFAULT_MODEL}  
  
REWRITE_MODEL = _env("REWRITE_MODEL", "gpt-5-mini")  
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-3-large")  
  
REASONING_EFFORT_MODELS = _csv_env("REASONING_EFFORT_MODELS", "gpt-5.1")  
ALLOWED_REASONING_EFFORT = {"low", "medium", "high"}  
  
DEFAULT_SYSTEM_PROMPT = _env(  
    "DEFAULT_SYSTEM_PROMPT",  
    "あなたは社内文書に基づいて回答するアシスタントです。根拠がある場合は必ず[n]形式で出典番号を付け、最後にSourcesを列挙してください。",  
)  
  
DEFAULT_SEARCH_INDEX = _env("DEFAULT_SEARCH_INDEX", "")  
ALLOWED_SEARCH_INDEXES = _csv_env("ALLOWED_SEARCH_INDEXES", DEFAULT_SEARCH_INDEX) or (  
    {DEFAULT_SEARCH_INDEX} if DEFAULT_SEARCH_INDEX else set()  
)  
  
DEFAULT_DOC_CONTAINER = _env("DEFAULT_DOC_CONTAINER", "")  
ALLOWED_DOC_CONTAINERS = _csv_env("ALLOWED_DOC_CONTAINERS", DEFAULT_DOC_CONTAINER) or (  
    {DEFAULT_DOC_CONTAINER} if DEFAULT_DOC_CONTAINER else set()  
)  
  
DEFAULT_REWRITE_QUERIES = _env_bool("DEFAULT_REWRITE_QUERIES", True)  
  
DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 15)  
MAX_TOP_K = _env_int("MAX_TOP_K", 30)  
  
MAX_SYSTEM_PROMPT_LEN = _env_int("MAX_SYSTEM_PROMPT_LEN", 8000)  
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 30)  
  
SAS_EXPIRY_MINUTES = _env_int("SAS_EXPIRY_MINUTES", 30)  
  
# Search tuning  
MAX_MULTIQUERY = _env_int("MAX_MULTIQUERY", 4)  
PER_SEARCH_TOP = _env_int("PER_SEARCH_TOP", 30)  
SEARCH_MAX_WORKERS = _env_int("SEARCH_MAX_WORKERS", 12)  
  
SEMANTIC_CONFIGURATION_NAME = _env("SEMANTIC_CONFIGURATION_NAME", "default")  
SEMANTIC_STRICTNESS = float(_env("SEMANTIC_STRICTNESS", "0.1"))  
  
# vector検索のデフォルト（省略時）  
DEFAULT_USE_VECTOR_SEARCH = _env_bool("DEFAULT_USE_VECTOR_SEARCH", True)  
  
# SAS返却のデフォルト（省略時）: ★デフォルト false  
DEFAULT_INCLUDE_SAS = _env_bool("DEFAULT_INCLUDE_SAS", False)  
  
# Search field mapping  
F_ID = _env("SEARCH_FIELD_ID", "id")  
F_CONTENT = _env("SEARCH_FIELD_CONTENT", "content")  
F_TITLE = _env("SEARCH_FIELD_TITLE", "title")  
F_FILEPATH = _env("SEARCH_FIELD_FILEPATH", "filepath")  
F_CHUNK_ID = _env("SEARCH_FIELD_CHUNK_ID", "chunk_id")  
F_VECTOR = _env("SEARCH_FIELD_VECTOR", "contentVector")  
  
# Azure endpoints (Entra ID; no keys)  
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT", "")  
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", "2024-10-21")  
  
AZURE_SEARCH_ENDPOINT = _env("AZURE_SEARCH_ENDPOINT", "")  
  
AZURE_STORAGE_ACCOUNT_URL = _env("AZURE_STORAGE_ACCOUNT_URL", "")  # https://<acct>.blob.core.windows.net  
  
# Embedding dimensions (must match Azure AI Search vector field configuration)  
EMBEDDING_DIMENSIONS = _env_int("EMBEDDING_DIMENSIONS", 1536)  
  
# -------------------------  
# Entra ID credential (local=Azure CLI, prod=Managed Identity)  
# -------------------------  
def build_credential():  
    if IS_LOCAL:  
        return AzureCliCredential()  
    mi_client_id = os.getenv("AZURE_CLIENT_ID")  
    return ManagedIdentityCredential(client_id=mi_client_id) if mi_client_id else ManagedIdentityCredential()  
  
  
credential = build_credential()  
  
  
def build_azure_ad_token_provider(credential_obj, scope: str):  
    def _provider() -> str:  
        return credential_obj.get_token(scope).token  
  
    return _provider  
  
  
# -------------------------  
# EasyAuth enforcement helpers  
# -------------------------  
def _is_true_env(name: str) -> bool:  
    v = (os.getenv(name) or "").strip().lower()  
    return v in ("1", "true", "yes", "y", "on")  
  
  
def is_easyauth_enabled_environment() -> bool:  
    """  
    Heuristic for Azure App Service EasyAuth being enabled.  
    If you run outside App Service, consider setting EASYAUTH_ENABLED=true explicitly.  
    """  
    if _is_true_env("EASYAUTH_ENABLED"):  
        return True  
    if _is_true_env("WEBSITE_AUTH_ENABLED"):  
        return True  
    if _is_true_env("APP_SERVICE_AUTH_ENABLED"):  
        return True  
    return False  
  
  
def enforce_easyauth_environment_or_raise() -> None:  
    if REQUIRE_EASYAUTH and (not IS_LOCAL) and (not is_easyauth_enabled_environment()):  
        raise RuntimeError("EasyAuth appears disabled but REQUIRE_EASYAUTH=true")  
  
  
def is_localhost_remote(req) -> bool:  
    return (req.remote_addr or "") in LOCALHOST_REMOTE_ADDRS  
  
  
# -------------------------  
# Auth (EasyAuth)  
# -------------------------  
def get_easy_auth_user(req) -> Optional[dict[str, str]]:  
    """  
    EasyAuth header X-MS-CLIENT-PRINCIPAL (base64 JSON)  
    returns: {"id": "...", "name": "..."} or None  
  
    Local dev bypass:  
      - only when APP_ENV=local  
      - only when req.remote_addr is localhost (127.0.0.1 / ::1)  
    """  
    # Enforce environment: in prod, require EasyAuth to be enabled at the platform level  
    if not IS_LOCAL:  
        try:  
            enforce_easyauth_environment_or_raise()  
        except Exception:  
            logger.exception("easyauth_environment_check_failed")  
            return None  
  
    cp = req.headers.get("X-MS-CLIENT-PRINCIPAL")  
    if cp:  
        try:  
            cp = cp + "=" * (-len(cp) % 4)  # padding  
            decoded = base64.b64decode(cp).decode("utf-8")  
            data = json.loads(decoded)  
  
            user_id, user_name = None, None  
            for c in data.get("claims", []):  
                typ = c.get("typ")  
                val = c.get("val")  
                if typ == "http://schemas.microsoft.com/identity/claims/objectidentifier":  
                    user_id = val  
                elif typ == "name":  
                    user_name = val  
  
            if user_id:  
                return {"id": user_id, "name": user_name or ""}  
        except Exception:  
            logger.exception("failed_to_parse_easyauth_header")  
            return None  
  
    # Local bypass (strict: only from localhost remote_addr)  
    if IS_LOCAL and is_localhost_remote(req):  
        return {"id": "localdev", "name": "localdev"}  
  
    return None  
  
  
# -------------------------  
# Azure clients (Entra ID)  
# -------------------------  
def get_openai_client() -> AzureOpenAI:  
    if not AZURE_OPENAI_ENDPOINT:  
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")  
  
    token_provider = build_azure_ad_token_provider(credential, "https://cognitiveservices.azure.com/.default")  
    return AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_version=AZURE_OPENAI_API_VERSION,  
        azure_ad_token_provider=token_provider,  
    )  
  
  
def get_search_client(index_name: str) -> SearchClient:  
    if not AZURE_SEARCH_ENDPOINT:  
        raise RuntimeError("AZURE_SEARCH_ENDPOINT is not set")  
  
    return SearchClient(  
        endpoint=AZURE_SEARCH_ENDPOINT,  
        index_name=index_name,  
        credential=credential,  # TokenCredential  
    )  
  
  
_blob_service_client: Optional[BlobServiceClient] = None  
_blob_service_client_lock = threading.Lock()  
  
  
def get_blob_service_client() -> BlobServiceClient:  
    global _blob_service_client  
    if not AZURE_STORAGE_ACCOUNT_URL:  
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_URL is not set")  
  
    if _blob_service_client is not None:  
        return _blob_service_client  
  
    with _blob_service_client_lock:  
        if _blob_service_client is None:  
            _blob_service_client = BlobServiceClient(account_url=AZURE_STORAGE_ACCOUNT_URL, credential=credential)  
    return _blob_service_client  
  
  
# -------------------------  
# User Delegation Key (UDK) cache  
# -------------------------  
_udk_lock = threading.Lock()  
_udk_cached_key = None  
_udk_cached_expiry: Optional[datetime] = None  
  
  
def get_user_delegation_key_cached(  
    bsc: BlobServiceClient,  
    *,  
    start_skew_minutes: int = 5,  
    refresh_skew_minutes: int = 2,  
) -> Any:  
    """  
    Cache User Delegation Key (UDK) until near expiry.  
    - start is set slightly in the past to avoid clock skew  
    - refresh if expiry is within refresh_skew_minutes  
    """  
    global _udk_cached_key, _udk_cached_expiry  
  
    now = datetime.now(timezone.utc)  
  
    with _udk_lock:  
        if _udk_cached_key is not None and _udk_cached_expiry is not None:  
            if now + timedelta(minutes=refresh_skew_minutes) < _udk_cached_expiry:  
                return _udk_cached_key  
  
        start = now - timedelta(minutes=start_skew_minutes)  
        expiry = now + timedelta(minutes=SAS_EXPIRY_MINUTES)  
  
        udk = bsc.get_user_delegation_key(start, expiry)  
        _udk_cached_key = udk  
        _udk_cached_expiry = expiry  
        return udk  
  
  
# -------------------------  
# SAS URL generation (User Delegation SAS)  
# -------------------------  
def build_blob_sas_url(container: str, blob_path: str) -> str:  
    """  
    Entra ID + User Delegation SAS.  
    Requires RBAC on Storage for managed identity / caller.  
    If it fails, returns "" (API still works; download_url becomes empty).  
  
    NOTE:  
      - Uses cached UDK to avoid per-blob UDK fetch.  
      - Uses bsc.url.rstrip('/') to avoid double slashes in output URL.  
    """  
    try:  
        bsc = get_blob_service_client()  
  
        udk = get_user_delegation_key_cached(bsc)  
  
        bc = bsc.get_blob_client(container=container, blob=blob_path)  
        sas = generate_blob_sas(  
            account_name=bc.account_name,  
            container_name=container,  
            blob_name=blob_path,  
            user_delegation_key=udk,  
            permission=BlobSasPermissions(read=True),  
            # start/expiry are already embedded in the UDK (service validates),  
            # but keeping explicit times in SAS is fine. We align with cached expiry.  
            start=datetime.now(timezone.utc) - timedelta(minutes=5),  
            expiry=datetime.now(timezone.utc) + timedelta(minutes=SAS_EXPIRY_MINUTES),  
        )  
  
        c = quote(container, safe="")  
        p = quote(blob_path, safe="/")  
        base_url = (bsc.url or "").rstrip("/")  
        return f"{base_url}/{c}/{p}?{sas}"  
    except Exception:  
        logger.exception("build_blob_sas_url_failed")  
        return ""  
  
  
# -------------------------  
# Request / validation  
# -------------------------  
@dataclass  
class RagRequest:  
    request_id: str  
    user: dict[str, str]  
    query: str  
    history: list[dict[str, str]]  
  
    model: str  
    system_prompt: str  
    reasoning_effort: Optional[str]  
  
    search_index: str  
    doc_container: str  
    top_k: int  
    rewrite_queries: bool  
    use_vector: bool  
    include_sas: bool  # ★追加（デフォルト false）  
  
  
def _bad_request(msg: str, *, request_id: Optional[str] = None):  
    payload = {"error": msg}  
    if request_id:  
        payload["request_id"] = request_id  
    return jsonify(payload), 400  
  
  
def _forbidden(msg: str, *, request_id: Optional[str] = None):  
    payload = {"error": msg}  
    if request_id:  
        payload["request_id"] = request_id  
    return jsonify(payload), 403  
  
  
def _server_error(msg: str, *, request_id: Optional[str] = None):  
    payload = {"error": msg}  
    if request_id:  
        payload["request_id"] = request_id  
    return jsonify(payload), 500  
  
  
def _get_request_id(req) -> str:  
    return req.headers.get("X-Request-Id") or str(uuid.uuid4())  
  
  
def _parse_json_body(req, request_id: str) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, int]]]:  
    try:  
        body = req.get_json(force=True)  
        if body is None:  
            body = {}  
        if not isinstance(body, dict):  
            return None, _bad_request("json_body_must_be_object", request_id=request_id)  
        return body, None  
    except BadRequest:  
        return None, _bad_request("invalid_json", request_id=request_id)  
  
  
def parse_rag_request(req, user: dict[str, str]) -> tuple[Optional[RagRequest], Optional[tuple[Any, int]]]:  
    request_id = _get_request_id(req)  
  
    body, err = _parse_json_body(req, request_id)  
    if err is not None:  
        return None, err  
    assert body is not None  
  
    query = (body.get("query") or "").strip()  
    if not query:  
        return None, _bad_request("missing_query", request_id=request_id)  
  
    history_in = body.get("history") or []  
    if not isinstance(history_in, list):  
        return None, _bad_request("history_must_be_list", request_id=request_id)  
  
    if MAX_HISTORY_MESSAGES <= 0:  
        history_in = []  
    else:  
        history_in = history_in[-MAX_HISTORY_MESSAGES:]  
  
    history: list[dict[str, str]] = []  
    for m in history_in:  
        if not isinstance(m, dict):  
            continue  
        role = (m.get("role") or "").strip()  
        content = m.get("content") or ""  
        if role not in ("user", "assistant"):  
            continue  
        history.append({"role": role, "content": str(content)})  
  
    model = (body.get("model") or DEFAULT_MODEL).strip()  
    if model not in ALLOWED_MODELS:  
        return None, _bad_request("invalid_model", request_id=request_id)  
  
    system_prompt = (body.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()  
    if len(system_prompt) > MAX_SYSTEM_PROMPT_LEN:  
        return None, _bad_request("system_prompt_too_long", request_id=request_id)  
  
    reasoning_effort = body.get("reasoning_effort")  
    if reasoning_effort is not None:  
        reasoning_effort = str(reasoning_effort).strip().lower()  
        if reasoning_effort == "":  
            reasoning_effort = None  
        elif reasoning_effort not in ALLOWED_REASONING_EFFORT:  
            return None, _bad_request("invalid_reasoning_effort", request_id=request_id)  
        elif model not in REASONING_EFFORT_MODELS:  
            reasoning_effort = None  
  
    search = body.get("search") or {}  
    if not isinstance(search, dict):  
        return None, _bad_request("search_must_be_object", request_id=request_id)  
  
    search_index = (search.get("index") or DEFAULT_SEARCH_INDEX).strip()  
    if not search_index:  
        return None, _bad_request("missing_search_index", request_id=request_id)  
    if ALLOWED_SEARCH_INDEXES and search_index not in ALLOWED_SEARCH_INDEXES:  
        return None, _forbidden("invalid_search_index", request_id=request_id)  
  
    doc_container = (search.get("container") or DEFAULT_DOC_CONTAINER).strip()  
    if not doc_container:  
        return None, _bad_request("missing_container", request_id=request_id)  
    if ALLOWED_DOC_CONTAINERS and doc_container not in ALLOWED_DOC_CONTAINERS:  
        return None, _forbidden("invalid_container", request_id=request_id)  
  
    rewrite_queries_flag = _bool(search.get("rewrite_queries"), DEFAULT_REWRITE_QUERIES)  
  
    # vector検索ON/OFF  
    use_vector_flag = _bool(search.get("use_vector"), DEFAULT_USE_VECTOR_SEARCH)  
  
    # ★SAS返却ON/OFF（デフォルト false）  
    include_sas_flag = _bool(search.get("include_sas"), DEFAULT_INCLUDE_SAS)  
  
    top_k = search.get("top_k", DEFAULT_TOP_K)  
    try:  
        top_k = int(top_k)  
    except Exception:  
        return None, _bad_request("top_k_must_be_int", request_id=request_id)  
    if top_k < 1 or top_k > MAX_TOP_K:  
        return None, _bad_request(f"top_k_out_of_range(1..{MAX_TOP_K})", request_id=request_id)  
  
    return (  
        RagRequest(  
            request_id=request_id,  
            user=user,  
            query=query,  
            history=history,  
            model=model,  
            system_prompt=system_prompt,  
            reasoning_effort=reasoning_effort,  
            search_index=search_index,  
            doc_container=doc_container,  
            top_k=top_k,  
            rewrite_queries=rewrite_queries_flag,  
            use_vector=use_vector_flag,  
            include_sas=include_sas_flag,  
        ),  
        None,  
    )  
  
  
# -------------------------  
# RAG core  
# -------------------------  
def _strip_code_fences(s: str) -> str:  
    s = s.strip()  
    if s.startswith("```"):  
        s = s.split("\n", 1)[-1]  
        if s.endswith("```"):  
            s = s.rsplit("```", 1)[0]  
    return s.strip()  
  
  
def rewrite_queries(openai_client: AzureOpenAI, query: str, max_n: int = 4) -> list[str]:  
    """Always includes original query as fallback."""  
    max_n = max(1, int(max_n))  
    try:  
        prompt = (  
            "次のユーザー質問を、検索に適した短いクエリに言い換えてください。\n"  
            f"- 最大{max_n}個\n"  
            '- 出力はJSON配列のみ（例: ["...","..."]）\n'  
            "- 同義語/略語/関連語も混ぜる\n\n"  
            f"質問: {query}"  
        )  
        resp = openai_client.chat.completions.create(  
            model=REWRITE_MODEL,  
            messages=[  
                {"role": "system", "content": "あなたは検索クエリ生成器です。余計な文章を出さずJSONだけを返します。"},  
                {"role": "user", "content": prompt},  
            ],  
        )  
        text = _strip_code_fences((resp.choices[0].message.content or "[]").strip())  
        arr = json.loads(text)  
        if isinstance(arr, list):  
            qs = [str(x).strip() for x in arr if str(x).strip()]  
            merged = [query] + [q for q in qs if q != query]  
            return merged[: min(max_n, MAX_MULTIQUERY)]  
    except Exception:  
        logger.exception("rewrite_queries_failed")  
    return [query]  
  
  
def embed_many(openai_client: AzureOpenAI, texts: list[str]) -> list[list[float]]:  
    """  
    Batch embeddings in ONE request:  
      embeddings.create(input=[q1,q2,...])  
    """  
    texts = [str(t) for t in texts]  
    resp = openai_client.embeddings.create(  
        model=EMBEDDING_MODEL,  
        input=texts,  
        dimensions=EMBEDDING_DIMENSIONS,  
    )  
    # SDK returns in the same order as input for a single request  
    return [d.embedding for d in resp.data]  
  
  
# -------------------------  
# Search (simple + semantic(rerank) + optional vector, RRF, de-dupe chunks)  
# -------------------------  
def _doc_key(hit: dict[str, Any]) -> str:  
    """チャンク重複排除のキー: filepath::chunk_id → chunk_id → id → filepath"""  
    chunk_id = str(hit.get("chunk_id") or "").strip()  
    doc_id = str(hit.get("id") or "").strip()  
    filepath = str(hit.get("filepath") or "").strip()  
  
    if filepath and chunk_id:  
        return f"{filepath}::{chunk_id}"  
    if chunk_id:  
        return chunk_id  
    if doc_id:  
        return doc_id  
    if filepath:  
        return filepath  
    return str(uuid.uuid4())  
  
  
def _map_hit(hit: Any) -> dict[str, Any]:  
    score = hit.get("@search.score", 0.0)  
    reranker = hit.get("@search.reranker_score", None)  
  
    doc_id = str(hit.get(F_ID) or "")  
    title = str(hit.get(F_TITLE) or "")  
    filepath = str(hit.get(F_FILEPATH) or "")  
    chunk_id = str(hit.get(F_CHUNK_ID) or "")  
    content = str(hit.get(F_CONTENT) or "")  
  
    item = {  
        "_key": "",  
        "score": float(score) if score is not None else 0.0,  
        "reranker_score": (float(reranker) if reranker is not None else None),  
        "id": doc_id,  
        "title": title,  
        "filepath": filepath,  
        "chunk_id": chunk_id,  
        "content": content,  
    }  
    item["_key"] = _doc_key(item)  
    return item  
  
  
def _better_item(a: dict[str, Any], b: dict[str, Any]) -> bool:  
    """a を残すべきなら True（semantic reranker_score 優先、次に score）"""  
    ar = a.get("reranker_score")  
    br = b.get("reranker_score")  
    if ar is not None or br is not None:  
        return (ar if ar is not None else -1e18) >= (br if br is not None else -1e18)  
    return (a.get("score") or 0.0) >= (b.get("score") or 0.0)  
  
  
def rrf_fuse(result_lists: list[list[dict[str, Any]]], top_k: int, k: int = 60) -> list[dict[str, Any]]:  
    scores: dict[str, float] = {}  
    best_item: dict[str, dict[str, Any]] = {}  
  
    for lst in result_lists:  
        for rank, item in enumerate(lst, start=1):  
            key = item["_key"]  
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)  
            if key not in best_item or not _better_item(best_item[key], item):  
                best_item[key] = item  
  
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]  
    out: list[dict[str, Any]] = []  
    for key, s in fused:  
        item = dict(best_item[key])  
        item["_rrf"] = s  
        out.append(item)  
    return out  
  
  
def _select_fields() -> str:  
    return ",".join(list({F_ID, F_CONTENT, F_TITLE, F_FILEPATH, F_CHUNK_ID}))  
  
  
def _search_simple(index_name: str, query: str, top: int) -> list[dict[str, Any]]:  
    sc = get_search_client(index_name)  
    it = sc.search(  
        search_text=query,  
        query_type="simple",  
        search_fields=[F_TITLE, F_CONTENT],  
        top=top,  
        select=_select_fields(),  
    )  
    return [_map_hit(x) for x in it]  
  
  
def _search_semantic(index_name: str, query: str, top: int, strictness: float) -> list[dict[str, Any]]:  
    sc = get_search_client(index_name)  
    it = sc.search(  
        search_text=query,  
        query_type="semantic",  
        semantic_configuration_name=SEMANTIC_CONFIGURATION_NAME,  
        search_fields=[F_TITLE, F_CONTENT],  
        top=top,  
        select=_select_fields(),  
    )  
    items = [_map_hit(x) for x in it]  
  
    out: list[dict[str, Any]] = []  
    for d in items:  
        s = d.get("reranker_score")  
        if s is None:  
            s = d.get("score", 0.0)  
        if float(s or 0.0) >= float(strictness):  
            out.append(d)  
    return out  
  
  
def _search_vector(index_name: str, vector: list[float], top: int) -> list[dict[str, Any]]:  
    sc = get_search_client(index_name)  
    vq = VectorizedQuery(vector=vector, k_nearest_neighbors=top, fields=F_VECTOR)  
    it = sc.search(  
        search_text="*",  
        vector_queries=[vq],  
        top=top,  
        select=_select_fields(),  
    )  
    return [_map_hit(x) for x in it]  
  
  
def search_hybrid_multiquery(  
    *,  
    index_name: str,  
    openai_client: AzureOpenAI,  
    queries: list[str],  
    fused_top: int,  
    strictness: float = SEMANTIC_STRICTNESS,  
    use_vector: bool = True,  
) -> list[dict[str, Any]]:  
    """  
    - queries[:MAX_MULTIQUERY]  
    - per query: simple + semantic (+ vector if enabled)  
    - RRF fuse  
    - de-dupe by chunk key  
    """  
    queries = [q.strip() for q in (queries or []) if str(q).strip()]  
    if not queries:  
        return []  
    queries = queries[:MAX_MULTIQUERY]  
  
    per_top = max(1, int(PER_SEARCH_TOP))  
    fused_top = max(1, int(fused_top))  
    use_vector = bool(use_vector)  
  
    vec_map: dict[str, list[float]] = {}  
    if use_vector:  
        try:  
            vectors = embed_many(openai_client, queries)  # ★batch  
            vec_map = {q: v for q, v in zip(queries, vectors)}  
        except Exception:  
            logger.exception("embed_many_failed_fallback_to_no_vector")  
            use_vector = False  
  
    result_lists: list[list[dict[str, Any]]] = []  
    futures: dict[Any, tuple[str, str]] = {}  
  
    t0 = time.perf_counter()  
    with ThreadPoolExecutor(max_workers=max(1, int(SEARCH_MAX_WORKERS))) as ex:  
        for q in queries:  
            futures[ex.submit(_search_simple, index_name, q, per_top)] = ("simple", q)  
            futures[ex.submit(_search_semantic, index_name, q, per_top, strictness)] = ("semantic", q)  
            if use_vector:  
                futures[ex.submit(_search_vector, index_name, vec_map[q], per_top)] = ("vector", q)  
  
        for fut in as_completed(futures):  
            stype, q = futures[fut]  
            try:  
                result_lists.append(fut.result())  
            except Exception:  
                logger.exception("search_task_failed type=%s query=%s", stype, q)  
                result_lists.append([])  
  
    fused = rrf_fuse(result_lists, top_k=fused_top, k=60)  
    t1 = time.perf_counter()  
    logger.info(  
        "search_hybrid_multiquery done queries=%d tasks=%d per_top=%d fused_top=%d use_vector=%s elapsed_ms=%d",  
        len(queries),  
        len(futures),  
        per_top,  
        fused_top,  
        str(use_vector).lower(),  
        int((t1 - t0) * 1000),  
    )  
    return fused  
  
  
def build_context_and_sources(  
    docs: list[dict[str, Any]],  
    container: str,  
    include_sas: bool,  
) -> tuple[str, list[dict[str, Any]]]:  
    sources: list[dict[str, Any]] = []  
    parts: list[str] = []  
  
    for i, d in enumerate(docs, start=1):  
        title = d.get("title") or (d.get("filepath") or "")  
        filepath = d.get("filepath") or ""  
        chunk_id = d.get("chunk_id") or ""  
        content = (d.get("content") or "").strip()  
  
        if len(content) > 2000:  
            content = content[:2000] + "…"  
  
        parts.append(  
            f"[{i}] title: {title}\n" f"path: {filepath}\n" f"chunk: {chunk_id}\n" f"text:\n{content}\n"  
        )  
  
        download_url = ""  
        if include_sas and filepath:  
            download_url = build_blob_sas_url(container, filepath)  
  
        sources.append(  
            {  
                "source_no": i,  
                "title": title,  
                "filepath": filepath,  
                "chunk_id": chunk_id,  
                "score": d.get("score", 0.0),  
                "reranker_score": d.get("reranker_score", None),  
                "rrf_score": d.get("_rrf", None),  
                "container": container,  
                "download_url": download_url,  
            }  
        )  
  
    context = "\n---\n".join(parts)  
    return context, sources  
  
  
def generate_answer_markdown(  
    openai_client: AzureOpenAI,  
    *,  
    model: str,  
    system_prompt: str,  
    history: list[dict[str, str]],  
    query: str,  
    context: str,  
    reasoning_effort: Optional[str],  
) -> str:  
    messages: list[dict[str, str]] = [  
        {"role": "system", "content": system_prompt},  
        {  
            "role": "system",  
            "content": "回答は必ず与えられたコンテキストに基づき、根拠がある文には出典番号[n]を付けてください。最後にSourcesを列挙してください。",  
        },  
    ]  
    messages.extend(history)  
  
    user_content = (  
        "以下は検索で見つかった社内資料の抜粋（Sources）です。\n"  
        "この情報のみを根拠として質問に回答してください。不明な場合は不明と述べてください。\n\n"  
        f"質問:\n{query}\n\n"  
        f"Sources context:\n{context}"  
    )  
    messages.append({"role": "user", "content": user_content})  
  
    kwargs: dict[str, Any] = {  
        "model": model,  
        "messages": messages,  
    }  
    if reasoning_effort:  
        kwargs["reasoning_effort"] = reasoning_effort  
  
    try:  
        resp = openai_client.chat.completions.create(**kwargs)  
    except TypeError:  
        if "reasoning_effort" in kwargs:  
            kwargs.pop("reasoning_effort", None)  
            resp = openai_client.chat.completions.create(**kwargs)  
        else:  
            raise  
  
    return (resp.choices[0].message.content or "").strip()  
  
  
def run_rag(req: RagRequest) -> dict[str, Any]:  
    t0 = time.perf_counter()  
  
    openai_client = get_openai_client()  
  
    tr0 = time.perf_counter()  
    used_queries = rewrite_queries(openai_client, req.query, max_n=MAX_MULTIQUERY) if req.rewrite_queries else [req.query]  
    tr1 = time.perf_counter()  
  
    ts0 = time.perf_counter()  
    docs = search_hybrid_multiquery(  
        index_name=req.search_index,  
        openai_client=openai_client,  
        queries=used_queries,  
        fused_top=req.top_k,  
        strictness=SEMANTIC_STRICTNESS,  
        use_vector=req.use_vector,  
    )  
    ts1 = time.perf_counter()  
  
    context, sources = build_context_and_sources(  
        docs=docs,  
        container=req.doc_container,  
        include_sas=req.include_sas,  # ★デフォルト false  
    )  
  
    tg0 = time.perf_counter()  
    answer = generate_answer_markdown(  
        openai_client=openai_client,  
        model=req.model,  
        system_prompt=req.system_prompt,  
        history=req.history,  
        query=req.query,  
        context=context,  
        reasoning_effort=req.reasoning_effort,  
    )  
    tg1 = time.perf_counter()  
  
    t1 = time.perf_counter()  
    return {  
        "request_id": req.request_id,  
        "answer": answer,  
        "used_queries": used_queries,  
        "sources": sources,  
        "user": {"id": req.user.get("id", ""), "name": req.user.get("name", "")},  
        "timing_ms": {  
            "rewrite": int((tr1 - tr0) * 1000),  
            "search": int((ts1 - ts0) * 1000),  
            "generate": int((tg1 - tg0) * 1000),  
            "total": int((t1 - t0) * 1000),  
        },  
        "config_effective": {  
            "model": req.model,  
            "reasoning_effort": req.reasoning_effort,  
            "search": {  
                "index": req.search_index,  
                "container": req.doc_container,  
                "top_k": req.top_k,  
                "rewrite_queries": req.rewrite_queries,  
                "use_vector": req.use_vector,  
                "include_sas": req.include_sas,  
                "max_multiquery": MAX_MULTIQUERY,  
                "per_search_top": PER_SEARCH_TOP,  
                "semantic_configuration_name": SEMANTIC_CONFIGURATION_NAME,  
                "semantic_strictness": SEMANTIC_STRICTNESS,  
            },  
            "auth": {  
                "require_easyauth": REQUIRE_EASYAUTH,  
                "app_env": APP_ENV,  
            },  
        },  
    }  
  
  
# -------------------------  
# Flask app  
# -------------------------  
app = Flask(__name__)  
  
  
@app.get("/healthz")  
def healthz():  
    return jsonify({"ok": True}), 200  
  
  
@app.post("/v1/rag")  
def rag_endpoint():  
    user = get_easy_auth_user(request)  
    if not user:  
        return Response("Unauthorized", status=401)  
  
    parsed, err = parse_rag_request(request, user)  
    if err is not None:  
        return err  
    assert parsed is not None  
  
    try:  
        result = run_rag(parsed)  
        return jsonify(result), 200  
    except Exception:  
        logger.exception("rag_failed request_id=%s", parsed.request_id)  
        return _server_error("internal_error", request_id=parsed.request_id)  
  
  
if __name__ == "__main__":  
    port = _env_int("PORT", 8000)  
    app.run(host="0.0.0.0", port=port, debug=IS_LOCAL)  