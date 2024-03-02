from langchain.docstore.document import Document
import pdftotext


class MyPDF:
    def __init__(self, pdfs):
        self.pdfs = pdfs

    def get_pdf_docs(self):
        docs = []
        for pdf in self.pdfs:
            pdf = pdftotext.PDF(pdf)
            i = 1
            for page in pdf:
                docs.append(Document(page_content=page, metadata={"page": i}))
                i += 1
        return docs
