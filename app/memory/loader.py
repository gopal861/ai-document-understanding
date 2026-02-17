# app/memory/loader.py

"""
Unified ingestion loader for engineering documentation.

Architecture contract preserved:
loader → chunker → embedder → vector_store

Supports bounded ingestion for:
- GitHub repositories (.md, .ipynb, .py)
- HTML documentation
- Markdown files
- PDF files
"""

from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import json


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
# SAFETY: CHARACTER LIMIT ENFORCEMENT
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

    text = "\n".join(text_parts)

    return enforce_character_limit(text)


# ============================================================
# MARKDOWN LOADER
# ============================================================

def load_markdown_text(file_path: str) -> str:

    with open(file_path, "r", encoding="utf-8") as f:

        text = f.read()

    return enforce_character_limit(text)


# ============================================================
# HTML DOCUMENTATION LOADER (BOUNDED)
# ============================================================

def load_html_docs_recursive(base_url: str) -> str:

    visited = set()

    queue = [base_url]

    collected_text = []

    while queue and len(visited) < MAX_HTML_PAGES:

        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:

            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            page_text = soup.get_text(separator="\n").strip()

            if page_text:
                collected_text.append(page_text)

            # discover internal links safely
            for link in soup.find_all("a", href=True):

                full_url = urljoin(base_url, link["href"])

                parsed = urlparse(full_url)

                if parsed.netloc != urlparse(base_url).netloc:
                    continue

                if full_url not in visited:
                    queue.append(full_url)

        except Exception:
            continue

    text = "\n\n".join(collected_text)

    return enforce_character_limit(text)


# ============================================================
# GITHUB DOCUMENTATION LOADER (PRODUCTION CORRECT)
# ============================================================

def load_github_repo_markdown(repo_url: str) -> str:
    """
    Loads engineering documentation from GitHub repositories.

    Supports:
    - .md files
    - .ipynb files (CRITICAL)
    - .py example files

    Returns unified plain text for embedding pipeline.
    """

    parts = repo_url.replace("https://github.com/", "").split("/")

    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL")

    owner = parts[0]
    repo = parts[1]

    headers = {}

    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    # --------------------------------------------------------
    # Get repository metadata
    # --------------------------------------------------------

    repo_api = f"https://api.github.com/repos/{owner}/{repo}"

    repo_resp = requests.get(repo_api, headers=headers)

    if repo_resp.status_code != 200:
        raise ValueError("Failed to fetch GitHub repo metadata")

    default_branch = repo_resp.json()["default_branch"]

    # --------------------------------------------------------
    # Get full repository tree
    # --------------------------------------------------------

    tree_api = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    )

    tree_resp = requests.get(tree_api, headers=headers)

    if tree_resp.status_code != 200:
        raise ValueError("Failed to fetch GitHub repo tree")

    tree = tree_resp.json()["tree"]

    collected_text = []

    allowed_extensions = (".md", ".ipynb", ".py")

    for item in tree:

        if item["type"] != "blob":
            continue

        path = item["path"]

        if not path.endswith(allowed_extensions):
            continue

        raw_url = (
            f"https://raw.githubusercontent.com/"
            f"{owner}/{repo}/{default_branch}/{path}"
        )

        file_resp = requests.get(raw_url)

        if file_resp.status_code != 200:
            continue

        # ----------------------------------------
        # Markdown and Python files
        # ----------------------------------------

        if path.endswith(".md") or path.endswith(".py"):

            collected_text.append(
                f"\n\nFILE: {path}\n\n{file_resp.text}"
            )

        # ----------------------------------------
        # Notebook files (CRITICAL FIX)
        # ----------------------------------------

        elif path.endswith(".ipynb"):

            try:

                notebook = json.loads(file_resp.text)

                notebook_text = []

                for cell in notebook.get("cells", []):

                    source = cell.get("source", [])

                    if isinstance(source, list):
                        notebook_text.append("".join(source))

                collected_text.append(
                    f"\n\nFILE: {path}\n\n" + "\n".join(notebook_text)
                )

            except Exception:
                continue

    full_text = "\n\n".join(collected_text) 

    


    return enforce_character_limit(full_text)


# ============================================================
# UNIFIED ENTRY POINT (CRITICAL CONTRACT)
# ============================================================

def load_text(source: str) -> str:

    # URL ingestion
    if source.startswith("http://") or source.startswith("https://"):

        validate_url(source)

        if "github.com" in source:
            return load_github_repo_markdown(source)

        return load_html_docs_recursive(source)

    # Markdown ingestion
    if source.lower().endswith(".md"):
        return load_markdown_text(source)

    # PDF ingestion
    if source.lower().endswith(".pdf"):
        return load_pdf_text(source)

    raise ValueError(f"Unsupported source type: {source}")
