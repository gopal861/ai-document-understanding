# app/memory/loader.py
from pypdf import PdfReader

def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages) 


