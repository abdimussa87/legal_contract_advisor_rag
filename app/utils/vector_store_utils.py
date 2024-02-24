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

    def get_retriever(self):
        # load from disk
        vector_store = Chroma(
            persist_directory="../data/chroma_db", embedding_function=self.embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        return retriever
