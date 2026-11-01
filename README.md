# PDF_RAG_Chatbot

# PDF RAG Chatbot

A chatbot that answers questions using only the content of a PDF file.

## How it works

1. Reads the text from a PDF.
2. Splits the text into chunks and turns each chunk into an embedding.
3. Stores the chunks in ChromaDB.
4. For each question, finds the closest chunk. If it's too far away, it says "I don't know based on this PDF."
5. Otherwise, sends the chunk and the question to Gemini, which answers using only that text.

## Setup

1. Install the libraries:
   pip install -r requirements.txt

2. Get a free API key from Google AI Studio.

3. Set the key in your terminal (PowerShell):
   $env:GEMINI_API_KEY="your-key-here"

4. Run the program:
   python app.py

## Example

Question: THE FOUNDATION OF NEXT-GEN COMPUTING
Answer: Based on the text, the foundation is the convergence of deep neural networks, specialized hardware accelerators, and massive parameter representations.

Question: what is java
Answer: I don't know based on this PDF.

## test_ai.py checks that your api keyworks


