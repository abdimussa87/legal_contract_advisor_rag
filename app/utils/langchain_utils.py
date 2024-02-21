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
        template = """ Answer the following question based only on the provided context:
        ### CONTEXT
        {context}

        ### User Prompt
        User Prompt: {user_prompt}
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
