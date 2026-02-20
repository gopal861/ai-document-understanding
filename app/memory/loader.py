# app/memory/loader.py

"""
Unified ingestion loader for engineering documentation.

Architecture contract preserved:
loader → chunker → embedder → vector_store

Supports:
- PDF files
- GitHub repositories
- Raw markdown URLs
- Static HTML docs (recursive crawl)
- Dynamic JS-rendered docs (Playwright fallback, Render-safe)
"""

from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os

from playwright.sync_api import sync_playwright

from app.config import (
    ALLOWED_DOMAINS,
    MAX_DOCUMENT_CHARACTERS,
    MAX_HTML_PAGES,
)

# ============================================================
# GLOBAL CONFIG (Render-safe Playwright path)
# ============================================================

PLAYWRIGHT_BROWSER_PATH = "/opt/render/project/.playwright"

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = PLAYWRIGHT_BROWSER_PATH


# ============================================================
# SAFETY: URL VALIDATION
# ============================================================

def validate_url(source: str):

    parsed = urlparse(source)

    if parsed.netloc not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain not allowed: {parsed.netloc}")


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

    parts = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            parts.append(text)

    return enforce_character_limit("\n".join(parts))


# ============================================================
# LOCAL MARKDOWN LOADER
# ============================================================

def load_markdown_text(file_path: str) -> str:

    with open(file_path, "r", encoding="utf-8") as f:
        return enforce_character_limit(f.read())


# ============================================================
# RAW MARKDOWN URL LOADER
# ============================================================

def load_raw_markdown_url(source: str) -> str:

    resp = requests.get(source, timeout=15)

    if resp.status_code != 200:
        raise ValueError("Failed to fetch markdown")

    return enforce_character_limit(resp.text)


# ============================================================
# CLEAN HTML → TEXT
# ============================================================

def extract_clean_text(html: str) -> str:

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript"
    ]):
        tag.decompose()

    return soup.get_text(separator="\n").strip()


# ============================================================
# STATIC HTML RECURSIVE CRAWLER (FAST PATH)
# ============================================================

def load_html_docs_recursive(base_url: str) -> str:

    visited = set()

    queue = [base_url]

    collected = []

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

            text = extract_clean_text(resp.text)

            if text and len(text) > 300:
                collected.append(text)

            soup = BeautifulSoup(resp.text, "html.parser")

            for link in soup.find_all("a", href=True):

                full_url = urljoin(base_url, link["href"])

                parsed = urlparse(full_url)

                if (
                    parsed.netloc == base_domain
                    and full_url not in visited
                    and full_url not in queue
                ):
                    queue.append(full_url)

        except Exception:
            continue

        if sum(len(x) for x in collected) > MAX_DOCUMENT_CHARACTERS:
            break

    return enforce_character_limit("\n\n".join(collected))


# ============================================================
# PLAYWRIGHT FALLBACK (RENDER SAFE)
# ============================================================

def load_dynamic_html_playwright(url: str) -> str:

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        page = browser.new_page()

        page.goto(url, timeout=30000)

        page.wait_for_timeout(2000)

        html = page.content()

        browser.close()

    return enforce_character_limit(
        extract_clean_text(html)
    )


# ============================================================
# GITHUB REPO LOADER
# ============================================================

def load_github_repo_markdown(repo_url: str) -> str:

    parts = repo_url.replace("https://github.com/", "").split("/")

    owner, repo = parts[0], parts[1]

    api = f"https://api.github.com/repos/{owner}/{repo}"

    repo_data = requests.get(api).json()

    branch = repo_data["default_branch"]

    tree_api = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/git/trees/{branch}?recursive=1"
    )

    tree = requests.get(tree_api).json()["tree"]

    collected = []

    for item in tree:

        if item["type"] != "blob":
            continue

        if not item["path"].endswith((".md", ".py", ".ipynb")):
            continue

        raw = (
            f"https://raw.githubusercontent.com/"
            f"{owner}/{repo}/{branch}/{item['path']}"
        )

        resp = requests.get(raw)

        if resp.status_code == 200:
            collected.append(resp.text)

        if sum(len(x) for x in collected) > MAX_DOCUMENT_CHARACTERS:
            break

    return enforce_character_limit("\n\n".join(collected))


# ============================================================
# MAIN ENTRY POINT (ARCHITECTURE CONTRACT)
# ============================================================

def load_text(source: str) -> str:

    if source.startswith("http://") or source.startswith("https://"):

        validate_url(source)

        domain = urlparse(source).netloc

        if domain == "raw.githubusercontent.com":
            return load_raw_markdown_url(source)

        if domain == "github.com":
            return load_github_repo_markdown(source)

        text = load_html_docs_recursive(source)

        # fallback if insufficient content
        if len(text) < 1000:

            dynamic_text = load_dynamic_html_playwright(source)

            if len(dynamic_text) > len(text):
                text = dynamic_text

        return enforce_character_limit(text)

    if source.lower().endswith(".pdf"):
        return load_pdf_text(source)

    if source.lower().endswith(".md"):
        return load_markdown_text(source)

    raise ValueError(f"Unsupported source: {source}")


