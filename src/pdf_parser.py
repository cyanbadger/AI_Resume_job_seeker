import fitz
import io

from PIL import Image

from src.ocr import extract_text_from_image


def extract_text_from_pdf(pdf_file):

    document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    extracted_text = ""

    # First attempt: normal PDF text extraction
    for page in document:

        text = page.get_text()

        extracted_text += text + "\n"

    # If text was successfully extracted
    if extracted_text.strip():

        document.close()

        return extracted_text.strip(), "PDF Text Extraction"

    # -----------------------------------------
    # Scanned PDF → OCR fallback
    # -----------------------------------------

    ocr_text = ""

    for page in document:

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        image_bytes = pixmap.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        page_text = extract_text_from_image(image)

        ocr_text += page_text + "\n"

    document.close()

    return ocr_text.strip(), "Tesseract OCR"


def extract_text_from_image_file(image_file):

    image = Image.open(image_file)

    text = extract_text_from_image(image)

    return text, "Tesseract OCR"


def extract_text_from_file(uploaded_file):

    file_type = uploaded_file.type

    if file_type == "application/pdf":

        return extract_text_from_pdf(uploaded_file)

    elif file_type in ["image/png", "image/jpeg"]:

        return extract_text_from_image_file(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file type."
        )