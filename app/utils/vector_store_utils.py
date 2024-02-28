from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class MyVectorStore:
    def __init__(self):
        model_name = "BAAI/bge-m3"
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity

        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            # model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs,
        )

    def embed_docs(self, chunks):
        db = FAISS.from_documents(chunks, self.embedding_function)
        db.save_local("../data/faiss_index")

    def get_retriever(self):
        vector_store = FAISS.load_local("../data/faiss_index", self.embedding_function)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        return retriever
