# RAG Based PDF Question Answering System

A simple RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and ask questions about them. 

## Features

- Upload any PDF document

- Ask questions about the content

- Get AI-powered answers based on the document

## Installation

1. Clone this repository

2. git clone https://github.com/Geetal28/PDF-Insight.git

3. cd PDF-Insight

4. Install dependencies

5. pip install -r requirements.txt

6. Set up your Groq API key

7. Get your free API key at: https://console.groq.com/keys

## Usage Run the application: python app.py

Open your browser and go to the displayed URL (usually http://127.0.0.1:7860)

# How It Works

1. Upload a PDF document

2. Click "Process PDF" and wait for confirmation

3. Ask questions about the document

4. Get AI-generated answers

# Tech Stack

 - LangChain - Document processing

- ChromaDB - Vector database

- SentenceTransformers - Embeddings

- Groq - LLM (Llama 3.1)

- Gradio - Web interface

# Project Structure

── rag_pipeline.py # Core RAG logic

── gradio_app.py # Web interface

── requirements.txt # Dependencies

── README.md
