import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import requests
from io import StringIO

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded
if not api_key:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

# Streamlit UI
st.title("Document Query")

input_type = st.selectbox("Choose input type", ["PDF file", "Text file", "URL"])

raw_text = ''

if input_type == "PDF file":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read PDF file
        pdfreader = PdfReader(uploaded_file)
        # Extract text from PDF
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

elif input_type == "Text file":
    uploaded_file = st.file_uploader("Choose a Text file", type="txt")
    if uploaded_file is not None:
        # Read text file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        raw_text = stringio.read()

elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        response = requests.get(url)
        if response.status_code == 200:
            raw_text = response.text
        else:
            st.error("Failed to fetch content from URL")

if raw_text:
    # Split text using character text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)

    # Create OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create FAISS vector store
    document_search = FAISS.from_texts(texts, embeddings)

    # Load QA chain
    chain = load_qa_chain(OpenAI(api_key=api_key), chain_type="stuff")

    # Query input
    query = st.text_input("Enter your query")

    if query:
        docs = document_search.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        # Display the response
        st.markdown("### Response")
        st.write(response)
