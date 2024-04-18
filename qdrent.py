
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


client = qdrant_client.QdrantClient(
    os.getenv('QDRENT_HOST'),
    api_key= os.getenv('QDRENT_API_KEY')
)

#create collection
os.environ['QDRANT_COLLECTION_NAME'] = 'my-collection'
vector_config = qdrant_client.http.models.VectorParams(size = 1536, distance = qdrant_client.http.models.Distance.COSINE)

client.recreate_collection(
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
    vectors_config=vector_config,
)


#create vector store 
#os.environ['OPEN_API_KEY'] =
embeddings = OpenAIEmbeddings()
vector_store = Qdrant(
    client=client, 
    collection_name=os.environ['QDRANT_COLLECTION_NAME'], 
    embeddings=embeddings,
)

#plug vector store into retrieval chain




qa = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}),
    chain_type= "stuff",
    retriever = vector_store.as_retriever()
)


def get_qdrent_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Qdrant.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



client = qdrant_client.QdrantClient(
    os.getenv('QDRENT_HOST'),
    api_key= os.getenv('QDRENT_API_KEY')
)

