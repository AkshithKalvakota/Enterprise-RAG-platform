import os
import pymupdf  # PyMuPDF for PDFs
from docx import Document  # python-docx for Word
from pptx import Presentation  # python-pptx for PowerPoint
import pandas as pd  # pandas for CSV/Excel

class DocumentParser:
    """
    A service class responsible for extracting raw text and metadata
    from various document formats including PDFs, Word, PPT, Excel, CSV, and TXT.

    Every parser returns:
        text      - the full extracted text
        metadata  - file-level info (file_type, source, ...)
        segments  - the text split by location, each {"text": ..., "location": {...}},
                    e.g. one segment per PDF page or slide, so chunks can cite where they came from
    """

    @staticmethod
    def parse(file_path: str) -> dict:
        """
        Detects file type by extension and delegates to the appropriate extraction method.
        Returns a dictionary containing raw text, metadata and segments.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path)
        elif ext == ".pptx":
            return DocumentParser._parse_pptx(file_path)
        elif ext == ".xlsx":
            return DocumentParser._parse_excel(file_path)
        elif ext == ".csv":
            return DocumentParser._parse_csv(file_path)
        elif ext in [".txt", ".md"]:
            return DocumentParser._parse_text(file_path)
        else:
            # Legacy .doc/.ppt/.xls are binary formats the libraries above cannot read
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _result(file_path: str, file_type: str, segments: list, **extra_metadata) -> dict:
        return {
            "text": "\n\n".join(segment["text"] for segment in segments),
            "metadata": {"file_type": file_type, "source": os.path.basename(file_path), **extra_metadata},
            "segments": segments,
        }

    @staticmethod
    def _parse_pdf(file_path: str) -> dict:
        segments = []
        with pymupdf.open(file_path) as doc:
            page_count = len(doc)
            for page_num in range(page_count):
                page_text = doc[page_num].get_text()
                if page_text.strip():
                    segments.append({"text": page_text, "location": {"page": page_num + 1}})
        return DocumentParser._result(file_path, "pdf", segments, page_count=page_count)

    @staticmethod
    def _parse_docx(file_path: str) -> dict:
        doc = Document(file_path)
        parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

        # Tables are not part of doc.paragraphs, so read them separately (one line per row)
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

        return DocumentParser._result(file_path, "docx", [{"text": "\n".join(parts), "location": {}}])

    @staticmethod
    def _parse_pptx(file_path: str) -> dict:
        prs = Presentation(file_path)
        segments = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_text = "\n".join(
                shape.text for shape in slide.shapes if shape.has_text_frame and shape.text.strip()
            )
            if slide_text:
                segments.append({"text": slide_text, "location": {"slide": slide_num}})
        return DocumentParser._result(file_path, "pptx", segments, slide_count=len(prs.slides))

    @staticmethod
    def _parse_excel(file_path: str) -> dict:
        # sheet_name=None reads every sheet, not just the first
        sheets = pd.read_excel(file_path, sheet_name=None)
        segments = [
            {"text": df.to_string(index=False), "location": {"sheet": sheet_name}}
            for sheet_name, df in sheets.items()
            if not df.empty
        ]
        return DocumentParser._result(file_path, "excel", segments, sheet_count=len(sheets))

    @staticmethod
    def _parse_csv(file_path: str) -> dict:
        df = pd.read_csv(file_path)
        return DocumentParser._result(file_path, "csv", [{"text": df.to_string(index=False), "location": {}}])

    @staticmethod
    def _parse_text(file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()
        return DocumentParser._result(file_path, "text", [{"text": text_content, "location": {}}])
