import pdfplumber

def pdf_to_text(file_obj) -> str:
    with pdfplumber.open(file_obj) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)
