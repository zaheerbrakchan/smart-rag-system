"""PDF text extraction (shared by upload and reindex)."""

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def extract_text_from_pdf(file_path: Path) -> List[Dict]:
    """
    Extract text from PDF with page numbers.
    Uses pdfplumber for better table extraction, falls back to pypdf.
    Near-empty pages are still returned here; use document_chunking.prepare_pages_for_indexing
    to drop them before embedding.
    """
    pages: List[Dict] = []

    try:
        import pdfplumber

        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = page.extract_tables()
                if tables:
                    table_text = ""
                    for table in tables:
                        for row in table:
                            row_text = " | ".join([str(cell) if cell else "" for cell in row])
                            table_text += row_text + "\n"
                    if table_text.strip():
                        text = (text or "") + "\n\n[TABLE DATA]\n" + table_text

                if text and text.strip():
                    pages.append({"text": text, "page_num": page_num + 1})

        if pages:
            print(f"✅ Extracted {len(pages)} pages using pdfplumber")
            return pages

    except ImportError:
        print("⚠️ pdfplumber not installed, falling back to pypdf")
    except Exception as e:
        print(f"⚠️ pdfplumber failed ({e}), falling back to pypdf")

    reader = PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page_num": page_num + 1})

    print(f"✅ Extracted {len(pages)} pages using pypdf")
    return pages
