import os
import streamlit as st
import time
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from Datachunksplitter import DataChunkSplitter
import fitz
import numpy as np
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

groq_api_key = os.environ["GROQ_API_KEY"]

# main_placeholder = st.empty()


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",  # Provide the pre-trained model's path
    model_kwargs={"device": "cpu"},  # Pass the model configuration options
    encode_kwargs={"normalize_embeddings": False},  # Pass the encoding options
)


# Initialize an instance of ChatGroq with the llama3-8b model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


# File upload for PDFs
st.title("PDF Embedding Generator")
uploaded_files = st.sidebar.file_uploader("Upload up to 3 PDF files", type="pdf", accept_multiple_files=True)

process_pdf_clicked = st.sidebar.button("Process PDFs")
main_placeholder = st.empty()


# Function to extract text from each PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

if process_pdf_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file to proceed.")
    else:
        #################################################
        # Data Loading Phase
        main_placeholder.text("Data Loading...Started...✅✅✅")
        pdf_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        
        #################################################
        # Data Splitting Phase
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        splitted_docs = []
        for text in pdf_texts:
            chunks = split_text_into_chunks(text)
            # Convert each chunk into a Document object
            splitted_docs.extend([Document(page_content=chunk) for chunk in chunks])

        ##############################################
        # VectorStore Phase
        main_placeholder.text("Embedding Vector Building...Started...✅✅✅")
        
        # Extract page content from Document objects to embed
        chunk_texts = [doc.page_content for doc in splitted_docs]
        
        # Generate embeddings
        embeddings = huggingface_embeddings.embed_documents(chunk_texts)
        
        # Combine texts and embeddings as pairs
        text_embeddings = list(zip(chunk_texts, embeddings))
        
        # Initialize FAISS index with text-embedding pairs
        vectorstore_faiss = FAISS.from_embeddings(text_embeddings, huggingface_embeddings)

        # Save FAISS index to local storage
        vectorstore_faiss.save_local("faiss_store")
        main_placeholder.text("FAISS Vectorstore Saved Successfully!")
        st.write("Embedding Vector Building Completed...✅✅✅")


##########################################33
#############################################
##Question Answering Phase
query = main_placeholder.text_input("Question: ")


## To only perform similarity search
# if query:
#     vectorstore = FAISS.load_local(
#         "faiss_store",
#         embeddings=huggingface_embeddings,
#         allow_dangerous_deserialization=True,
#     )
#     docs = vectorstore.similarity_search(query)
#     st.header("Answer")
#     st.write(docs[0].page_content)


if query:
    vectorstore = FAISS.load_local(
        "faiss_store",
        embeddings=huggingface_embeddings,
        allow_dangerous_deserialization=True,
    )
    print("dimention of loaded faiss", vectorstore.index.d)
    print("Vectorstore is loaded from file.")
    retriever = vectorstore.as_retriever(
        # search_type="similarity", search_kwargs={"k": 3}
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with few sentences maximum.

    {context}

    Question: {input}

    Helpful Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["input", "context"]
    )

    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    result = retrieval_chain.invoke({"input": query})
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", None)
    # sources = [doc.metadata["source"] for doc in docs]
    if sources:
        st.subheader("Sources:")
        # sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources:
            st.write(source)
