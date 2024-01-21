import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Helloo!! Ask Me anything"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!!"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_history(chain):
    with st.container():
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Ask about your documents:")
            submit_button = st.form_submit_button('Send')

        if submit_button and user_input:
            with st.spinner('Generating Response ....'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style='adventurer')
            message(st.session_state["generated"][i], key=str(i), avatar_style='fun-emoji')

def main():
    initialize_session_state()
    st.title('Multi-Docs Chatbot')

    st.sidebar.title('Document Processing')
    uploaded_files = st.sidebar.file_uploader('Upload files', accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in {".docx", ".doc"}:
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        llm = replicate.Replicate(streaming=True, model="mistralai/mistral-7b-instruct-v0.1:5fe0a3d7ac2852264a25279d1dfb798acbc4d49711d126646594e212cb821749",
                            model_kwargs={"temperature": 0.7, "max_length": 1000, "top_1": 1})

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                     memory=memory)

        display_history(chain)

if __name__ == '__main__':
    main()
