import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="++ RAG CHATBOT", page_icon="©️", layout="wide")
st.title("©️++ RAG CHATBOT")
st.write("Ask any question about C++ and get accurate answers!")

# Load environment variables
load_dotenv()

#cache document loading
@st.cache_resource
def load_vector_store():
    #load documents
    loader=TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents=loader.load()

    #split text into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents=text_splitter.split_documents(documents)

    #Embeddings
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #create FAISS Vector store
    db=FAISS.from_documents(final_documents, embeddings)
    return db

#vector database runs only once because of cache concepts
db=load_vector_store()

llm=Ollama(model="gemma2:2b")

#chat interface
text_input=st.text_input("Ask a question about C++:")
if text_input:
    with st.spinner("Processing your question..."):
        docs=db.similarity_search(text_input)
        context="\n".join([doc.page_content for doc in docs])
    prompt=f"Answer the following question based on the context provided:\n\nContext: {context}\n\nQuestion: {text_input}\n\nAnswer:"
    response=llm.invoke(prompt)
    st.subheader("Answer:")
    st.write(response)