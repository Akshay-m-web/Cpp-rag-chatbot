import streamlit as st
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Step 1:Page configuration
st.set_page_config(page_title="++ RAG CHATBOT", page_icon="©️")
st.title("©️++ RAG CHATBOT")
st.write("Ask any question about C++ and get accurate answers!")

#Step 2: Load environment variables
load_dotenv()

#step 3: cache document loading
@st.cache_resource
def load_vector_store():
    #step A: Load documents
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    #step B:split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=20
        #20 characters overlap it healps to maintain context connectivity between chunks
    )
    final_documents = text_splitter.split_documents(documents)

    #step C:Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        #this is embedding model that converts text into vector representation
    )

    #step D:create FAISS Vector store
    #converts each chunk of text into vector representation and stores it in a FAISS index for efficient similarity search
    db = FAISS.from_documents(final_documents, embeddings)

    return db

#vector database runs only once because of cache concepts
db=load_vector_store()

#User input
query = st.text_input("Enter your question about C++:")

if query:
    #covert user questions to embeddings
    #searches FAISS database
    #returns top 3 most relevant chunks 
    docs = db.similarity_search(query, k=3)
    st.subheader("📒 Retrieved Context")
    for i,doc in enumerate(docs):
        st.markdown(f"**Result {i+1} :**")
        st.write(doc.page_content)

