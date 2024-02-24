from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)


class MyLangChain:
    def generate_answer_chain(self, base_retriever):
        template = """Act as a legal contract answering expert. You will be presented with a legal contract as context and a question related to that contract. Your task is to provide a succinct answer to the question based on the content of the contract. Make sure you reply with "I don't know" if the answer cannot be found in the context.
        ### CONTEXT
        {context}

        ### Question
        Question: {user_prompt}
        """

        prompt = ChatPromptTemplate.from_template(template)

        primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        retriever = RunnableParallel(
            {
                "context": itemgetter("user_prompt") | base_retriever,
                "user_prompt": itemgetter("user_prompt"),
            }
        )

        retrieval_augmented_qa_chain = retriever | {
            "response": prompt | primary_qa_llm,
            "context": itemgetter("context"),
        }
        return retrieval_augmented_qa_chain
