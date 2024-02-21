from PyPDF2 import PdfReader


class MyPDF:
    def __init__(self, pdf):
        self.pdf = pdf

    def get_pdf_text(self):
        text = ""
        for pdf in self.pdf:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
