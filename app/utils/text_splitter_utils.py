from langchain.text_splitter import RecursiveCharacterTextSplitter


class MyTextSplitter:
    def __init__(self, text):
        self.text = text

    def get_text_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(self.text)
        return chunks
