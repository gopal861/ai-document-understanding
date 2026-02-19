# app/memory/loader.py

"""
Unified ingestion loader for engineering documentation.

Architecture contract preserved:
loader → chunker → embedder → vector_store

Supports:
- PDF files
- GitHub repositories
- Raw markdown URLs
- Static HTML docs
- Dynamic JS-rendered docs (Playwright fallback)
"""

from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import json

from playwright.sync_api import sync_playwright

from app.config import (
    ALLOWED_DOMAINS,
    MAX_DOCUMENT_CHARACTERS,
    MAX_HTML_PAGES,
)


# ============================================================
# SAFETY: URL VALIDATION
# ============================================================

def validate_url(source: str):

    parsed = urlparse(source)
    domain = parsed.netloc

    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain not allowed: {domain}")


# ============================================================
# SAFETY: CHARACTER LIMIT
# ============================================================

def enforce_character_limit(text: str) -> str:

    if len(text) > MAX_DOCUMENT_CHARACTERS:
        return text[:MAX_DOCUMENT_CHARACTERS]

    return text


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf_text(file_path: str) -> str:

    reader = PdfReader(file_path)

    text_parts = []

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text_parts.append(extracted)

    return enforce_character_limit("\n".join(text_parts))


# ============================================================
# LOCAL MARKDOWN LOADER
# ============================================================

def load_markdown_text(file_path: str) -> str:

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return enforce_character_limit(text)


# ============================================================
# RAW MARKDOWN URL LOADER (FAST PATH)
# ============================================================

def load_raw_markdown_url(source: str) -> str:

    resp = requests.get(source, timeout=15)

    if resp.status_code != 200:
        raise ValueError("Failed to fetch raw markdown")

    return enforce_character_limit(resp.text)


# ============================================================
# STATIC HTML LOADER
# ============================================================

def load_html_docs_recursive(base_url: str) -> str:

    visited = set()
    queue = [base_url]
    collected_text = []

    base_domain = urlparse(base_url).netloc

    while queue and len(visited) < MAX_HTML_PAGES:

        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:

            resp = requests.get(url, timeout=15)

            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n").strip()

            if text:
                collected_text.append(text)

            for link in soup.find_all("a", href=True):

                full_url = urljoin(base_url, link["href"])
                parsed = urlparse(full_url)

                if parsed.netloc == base_domain and full_url not in visited:
                    queue.append(full_url)

        except Exception:
            continue

    full_text = "\n\n".join(collected_text)

    return enforce_character_limit(full_text)


# ============================================================
# PLAYWRIGHT DYNAMIC DOC LOADER (NEW SAFE FALLBACK)
# ============================================================

def load_dynamic_html_playwright(url: str) -> str:
    """
    Used only when static HTML extraction fails.

    Safe for Render Free and HuggingFace:
    - launches browser lazily
    - closes immediately
    - respects character limits
    """

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(url, timeout=30000)

        page.wait_for_timeout(2000)

        content = page.content()

        browser.close()

    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n").strip()

    return enforce_character_limit(text)


# ============================================================
# GITHUB REPO LOADER
# ============================================================

def load_github_repo_markdown(repo_url: str) -> str:

    parts = repo_url.replace("https://github.com/", "").split("/")

    owner = parts[0]
    repo = parts[1]

    api = f"https://api.github.com/repos/{owner}/{repo}"

    repo_data = requests.get(api).json()

    branch = repo_data["default_branch"]

    tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

    tree = requests.get(tree_api).json()["tree"]

    collected = []

    for item in tree:

        if item["type"] != "blob":
            continue

        if not item["path"].endswith((".md", ".py", ".ipynb")):
            continue

        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}"

        resp = requests.get(raw)

        if resp.status_code == 200:
            collected.append(resp.text)

    return enforce_character_limit("\n\n".join(collected))


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def load_text(source: str) -> str:

    if source.startswith("http://") or source.startswith("https://"):

        validate_url(source)

        domain = urlparse(source).netloc

        if domain == "raw.githubusercontent.com":
            return load_raw_markdown_url(source)

        if domain == "github.com":
            return load_github_repo_markdown(source)

        # try fast static loader first
        text = load_html_docs_recursive(source)

        # fallback to Playwright if empty
        if not text or len(text.strip()) < 500:
            text = load_dynamic_html_playwright(source)

        return text

    if source.lower().endswith(".pdf"):
        return load_pdf_text(source)

    if source.lower().endswith(".md"):
        return load_markdown_text(source)

    raise ValueError(f"Unsupported source: {source}")

