from PyPDF2 import PdfReader
from langchain.docstore.document import Document


class MyPDF:
    def __init__(self, pdf):
        self.pdf = pdf

    def get_pdf_docs(self):
        docs = []
        for pdf in self.pdf:
            pdf_reader = PdfReader(pdf)
            i = 1
            for page in pdf_reader.pages:
                docs.append(
                    Document(
                        page_content=page.extract_text().replace("\n", " "),
                        metadata={"page": i},
                    )
                )
                i += 1
        return docs
