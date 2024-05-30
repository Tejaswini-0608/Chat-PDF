# Chat-PDF
PDF Text Extractor and QA System

This repository contains a Streamlit application that extracts text from PDF documents, processes the text into manageable chunks, and leverages Google's Generative AI to provide question-answering capabilities. The project utilizes Langchain for text processing and FAISS for vector storage, integrating advanced AI embeddings to enhance the QA system.

# Features:

  1. PDF Text Extraction: Upload multiple PDF documents and extract their textual content.
  
  2. Text Chunking: Split the extracted text into smaller, manageable chunks for efficient processing.
  
  3. AI Embeddings: Use Google Generative AI to generate embeddings for the text chunks.
  
  4. Vector Storage: Store embeddings in a FAISS vector store for efficient retrieval.
  
  5. Question-Answering: Implement a QA system using Langchain's question-answering chain to respond to user queries based on 
    the extracted and processed text.

# Technologies Used:
->Streamlit: A framework for creating interactive web applications.

->PyPDF2: A library for reading PDF files and extracting text.

->Langchain: Used for text splitting and QA chain implementation.

->Google Generative AI: Provides embeddings and chat capabilities.

->FAISS: A library for efficient similarity search and clustering of dense vectors.

->Python: The primary programming language for the project.

# Use case diagram:



![WhatsApp Image 2024-05-24 at 14 40 52](https://github.com/Tejaswini-0608/Chat-PDF/assets/100186885/f20a3263-586a-42f8-8d84-b1d7821dca3f)
