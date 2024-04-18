from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import  Qdrant
from langchain.embeddings import OpenAIEmbeddings
import qdrant_client
import streamlit as st
import os
from langchain.llms import OpenAI, HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
from langchain.chains import RetrievalQA
import urllib.parse


#initialzation of items in sasion state
if "conversation" not in st.session_state:
          st.session_state.conversation = []
if "chat_history" not in st.session_state:
          st.session_state.chat_history = []

parsed_url = urllib.parse.urlparse(os.getenv('QDRANT_HOST'))
host = parsed_url.hostname
port = parsed_url.port

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents.

    Args:
        pdf_docs (list): List of PDF documents.

    Returns:
        str: Concatenated text extracted from the PDFs.
    """
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks.

    Args:
        text (str): Input text.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_from_pdf(text_chunks):
    """Create a vector store from text chunks extracted from PDF.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        Qdrant: Vector store.
    """
    parsed_url = urllib.parse.urlparse(os.getenv('QDRANT_HOST'))
    host = parsed_url.hostname
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant.from_texts(
        text_chunks,
        embedding=embeddings,
        host = host,
        api_key = os.getenv("QDRANT_API_KEY"),
        collection_name=os.environ['QDRANT_COLLECTION_NAME']
    )

    return vector_store

def get_conversation_chain(vector_store):
    """Create a conversational chain with an OpenAI language model.

    Args:
        vector_store (Qdrant): Vector store.

    Returns:
        ConversationalRetrievalChain: Conversational retrieval chain.
    """
    llm = ChatOpenAI(model_name='gpt-4-turbo-preview') #model_name='gpt-4-turbo-preview'
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    """Handle user input in the conversation.

    Args:
        user_question (str): User's question.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def get_from_vector_store_qdrant():
    """Get a Qdrant vector store.

    Returns:
        Qdrant: Vector store.
    """
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )
    return vector_store
