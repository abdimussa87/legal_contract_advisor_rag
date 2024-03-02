import streamlit as st
from dotenv import load_dotenv
from utils.pdf_utils import MyPDF
from utils.text_splitter_utils import MyTextSplitter
from utils.vector_store_utils import MyVectorStore
from utils.langchain_utils import MyLangChain
from langchain_core.messages import AIMessage, HumanMessage


def handle_userinput(user_question):
    # a path which may happen if the document has already been embedded
    if not st.session_state.conv_chain:
        st.error(f"Please enter your document(s)")
        return

    result = st.session_state.conv_chain.invoke(
        {
            "chat_history": st.session_state.chat_history,
            "input": user_question,
        }
    )

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))


def setup_initial_session_state():
    # conv_chain session state
    if "conv_chain" not in st.session_state:
        st.session_state.conv_chain = None

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]


def main():
    # load environment variables
    load_dotenv(override=True)

    st.set_page_config(page_title="Contract Ai", page_icon="🤖")

    setup_initial_session_state()

    st.header("Answer anything relating to your contract")
    user_question = st.text_input("What is your question?")
    if user_question:
        handle_userinput(user_question)

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # initialize pdf reader
                pdf_reader = MyPDF(pdfs=pdf_docs)
                text_splitter = MyTextSplitter()
                # create vector store
                my_vector_store = MyVectorStore()

                my_lang_chain = MyLangChain(pdf_reader, text_splitter, my_vector_store)
                docs = my_lang_chain.setup_pdf_reader()

                # get the text chunks
                text_chunks = my_lang_chain.setup_text_splitter(docs)

                # embed the text chunks
                my_lang_chain.setup_vector_store(text_chunks)

                # get the retriever
                conversation_aware_retriever = (
                    my_lang_chain.setup_conversation_aware_retriever()
                )

                # create conversation_chain
                st.session_state.conv_chain = my_lang_chain.get_conversational_chain(
                    conversation_aware_retriever
                )


if __name__ == "__main__":
    main()
