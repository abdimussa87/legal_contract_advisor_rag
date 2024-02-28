from langchain.text_splitter import RecursiveCharacterTextSplitter


class MyTextSplitter:
    def __init__(self, docs):
        self.docs = docs

    def get_text_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.docs)
        return chunks
