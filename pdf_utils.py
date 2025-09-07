from __future__ import annotations

import os
from typing import List
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
	if not os.path.exists(pdf_path):
		raise FileNotFoundError(f"PDF not found: {pdf_path}")
	with open(pdf_path, "rb") as f:
		reader = PyPDF2.PdfReader(f)
		texts: List[str] = []
		for idx, page in enumerate(reader.pages):
			try:
				texts.append(page.extract_text() or "")
			except Exception:
				continue
	text = "\n".join(t.strip() for t in texts if t and t.strip())
	if not text:
		raise ValueError("No extractable text found in PDF")
	return text



