import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
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

# Function to save uploaded file
def save_uploaded_file(file, filename):
  # Create data folder if it doesn't exist
  if not os.path.exists("data"):
    os.makedirs("data")
  # Save the file to data folder with the original filename
  with open(os.path.join("data", filename), "wb") as f:
    f.write(file.read())

# Function to process text from a file
def process_text_file(filepath):
  with open(filepath, 'r', encoding="latin-1") as f:
    content = f.read()
  return content

# Streamlit UI
st.title("Document Query")

input_type = st.selectbox("Choose input type", ["PDF file", "Text file", "URL", "Existing Files"])

raw_text = ''
documents = []  # List to store all processed documents

if input_type == "PDF file":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read PDF file
        pdfreader = PdfReader(uploaded_file)
        # Extract text from PDF
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                documents.append(content)  # Append each page's content separately
        # Save uploaded file (optional)
        save_uploaded_file(uploaded_file, uploaded_file.name)


elif input_type == "Text file":
    uploaded_file = st.file_uploader("Choose a Text file", type="txt")
    if uploaded_file is not None:
        # Read text file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        raw_text = stringio.read()
        documents.append(raw_text)
        # Save uploaded file
        save_uploaded_file(uploaded_file, uploaded_file.name)

elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        response = requests.get(url)
        if response.status_code == 200:
            raw_text = response.text
            documents.append(raw_text)
        else:
            st.error("Failed to fetch content from URL")

elif input_type == "Existing Files":
  # Get list of files in data folder
  data_files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
  if data_files:
    selected_files = st.multiselect("Select files from data folder", data_files)
    for filename in selected_files:
      filepath = os.path.join("data", filename)
      documents.append(process_text_file(filepath))  # Call the process function
  else:
    st.info("No files found in data folder")

# Process documents (if any)
if documents:
  # Split text using character text splitter
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=800,
      chunk_overlap=200,
      length_function=len,
  )

  all_texts = []
  for doc in documents:
    all_texts.extend(text_splitter.split_text(doc))

  # Create OpenAIEmbeddings object
  embeddings = OpenAIEmbeddings(openai_api_key=api_key)

  # Create FAISS vector store
  document_search = FAISS.from_texts(all_texts, embeddings)

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
