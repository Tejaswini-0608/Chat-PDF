import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set Streamlit page config
st.set_page_config(page_title="Chat with PDF using Gemini💁", page_icon="💁", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* General settings */
        body {
            background-color: #f7f9fc;
            color: #333333;
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f7f9fc;
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #003366;
        }
        h1 {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        /* Button styling */
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            margin: 0.5rem 0;
        }
        .stButton>button:hover {
            background-color: #002244;
        }
        /* Input box styling */
        .stTextInput>div>div>input {
            border: 2px solid #003366;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 1rem;
        }
        /* Title and header styling */
        .st-title {
            color: #003366;
            text-align: center;
        }
        .st-header {
            color: #003366;
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into chunks suitable for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a vector store from text chunks using Google Generative AI embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the conversational AI chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, "answer is not available in the context". Don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Process user input, perform similarity search, and return the AI's response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    """Main function to run the Streamlit app."""
    st.title("Chat with PDF using Gemini💁")
    
    st.header("Ask Your PDF a Question")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete")

if __name__ == "__main__":
    main()
