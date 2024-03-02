from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever

    def get_parent_document_retriever(self, parent_splitter, child_splitter, base_docs):

        vectorstore = FAISS.load_local("../data/faiss_index", self.embedding_function)
        store = InMemoryStore()

        parent_document_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        parent_document_retriever.add_documents(base_docs)

        return parent_document_retriever

    def get_history_aware_retriever(self, llm, retriever):
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain
