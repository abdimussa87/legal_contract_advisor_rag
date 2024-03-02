from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class MyLangChain:

    def __init__(self) -> None:

        # setting up LangSmith observability
        self.setup_lang_smith()

        template = """Answer the question (by giving reference to the context you used) based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        local_llm = "mistral:instruct"
        self.llm = ChatOllama(model=local_llm)

    def generate_answer_chain(self, retriever):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def get_conversational_chain(self, retriever_chain):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question (by giving reference to the context you used) based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        stuff_documents_chain = create_stuff_documents_chain(self.llm, prompt)

        conversation_rag_chain = create_retrieval_chain(
            retriever_chain, stuff_documents_chain
        )
        return conversation_rag_chain

    def setup_lang_smith(self):
        load_dotenv(override=True)

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
