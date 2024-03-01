import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template

from utils.pdf_utils import MyPDF
from utils.text_splitter_utils import MyTextSplitter
from utils.vector_store_utils import MyVectorStore
from utils.langchain_utils import MyLangChain

load_dotenv(override=True)


def get_conversation_chain(retriever):
    my_lang_chain = MyLangChain()
    return my_lang_chain.generate_answer_chain(retriever)


def handle_userinput(user_question):
    # a path which may happen if the document has already been embedded
    if not st.session_state.conversation:
        my_vector_store = MyVectorStore()
        try:
            retriever = my_vector_store.get_retriever()
        except Exception:
            st.error(f"Please enter document")
            return

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(retriever)

    answer = st.session_state.conversation.invoke(
        user_question,
    )

    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Contract Ai", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Answer anything relating to your contract")
    user_question = st.text_input("What is your question?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                pdf = MyPDF(pdf=pdf_docs)
                docs = pdf.get_pdf_docs()

                # get the text chunks
                text_splitter = MyTextSplitter(docs)
                text_chunks = text_splitter.get_text_chunks()

                # create vector store
                my_vector_store = MyVectorStore()
                my_vector_store.embed_docs(text_chunks)

                # retriever = my_vector_store.get_retriever()

                retriever = my_vector_store.get_parent_document_retriever(
                    parent_splitter=text_splitter.get_parent_splitter(),
                    child_splitter=text_splitter.get_child_splitter(),
                    base_docs=text_chunks,
                )

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(retriever)


if __name__ == "__main__":
    main()
