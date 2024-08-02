# Document Query App

## Overview

The Document Query App is a Streamlit application that allows users to upload PDF files, text files, or provide URLs to extract and query textual content. The app uses the LangChain framework for text processing and the OpenAI API for querying and generating responses.

## Features

- **File Upload**: Users can upload PDF files or text files.
- **URL Input**: Users can provide a URL from which the app will fetch and extract text.
- **Querying**: Users can input a query to search the extracted text and receive responses based on the provided content.

## Requirements

To run this application, you'll need:

- Python 3.8 or higher
- The following Python packages:
  - `streamlit`
  - `PyPDF2`
  - `langchain-openai`
  - `langchain-community`
  - `typing-extensions`
  - `python-dotenv`
  - `faiss-cpu`
  - `requests`

You can install the required packages using pip:

```bash
pip install streamlit PyPDF2 langchain-openai langchain-community typing-extensions python-dotenv faiss-cpu requests
```
## Setup

#### 1. Clone the repository: 

```bash
git clone <https://github.com/your_username/Document-Query-Q-A.git>
cd <Document-Query-Q-A>
```

#### 2. Create a .env file in the project directory with the following content:

```bash
OPENAI_API_KEY=your-openai-api-key
```
Replace your-openai-api-key with your actual OpenAI API key.

#### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

#### 4. Run the Streamlit app:
```bash 
streamlit run app.py
```

This will start the Streamlit server, and you can access the app in your browser at http://localhost:8501.

## Usage :-
#### 1. Choose Input Type:

- Select from PDF file, Text file, or URL.
- If you select PDF file or Text file, use the file uploader to upload your document.
- If you select URL, enter the URL of the page containing the text you want to query.

![Image](Images/select_image.png)

![Image](Images/browse_file.png)

#### 2. Enter Query:

- Input your query in the provided text box and submit it to get the response based on the content extracted from the document or URL.

![Image](Images/enter_query.png)

#### 3. View Response:

- The app will display the response to your query on the same page.

![Image](Images/view_response.png)

## Troubleshooting :-

- Ensure that your .env file contains a valid OpenAI API key.
- Make sure you have an active internet connection, as the app needs to fetch and query data from OpenAI.

## Contributing :-

- Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

