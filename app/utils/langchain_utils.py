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


class MyLangChain:

    def __init__(self) -> None:
        self.prompt = PromptTemplate(
            template="""Act as a legal contract answering expert. You will be presented with a legal contract as context and a question related to that contract. Your task is to provide a succinct answer to the question based on the content of the contract. Make sure you reply with "I don't know" if the answer cannot be found in the context.
            ### CONTEXT
            {context}

            ### Question
            Question: {question}""",
            input_variables=["context", "question"],
        )

        local_llm = "mistral:instruct"
        self.llm = ChatOllama(model=local_llm, temperature=0)

    def generate_answer_chain(self, retriever):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
