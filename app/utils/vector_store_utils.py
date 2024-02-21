from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class MyVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        pass

    def embed_text(self, text_chunks):
        Chroma.from_texts(
            text_chunks, self.embeddings, persist_directory="../data/chroma_db"
        )
