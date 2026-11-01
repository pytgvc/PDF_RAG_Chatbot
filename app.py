from pypdf import PdfReader
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("my_pdf")

pdf_file = input("Enter the PDF file path: ")

reader = PdfReader(pdf_file)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

if not text.strip():
    print("No readable text found in this PDF.")

else:
    
    chunks = [
        text[i:i + 500]
        for i in range(0, len(text), 500)
    ]

    df = pd.DataFrame({
        "chunk_id": range(len(chunks)),
        "text": chunks
    })

    print("Total chunks:", len(df))

    embeddings = model.encode(chunks).tolist()

    collection.upsert(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print("PDF processed successfully!")

    while True:
        user_question = input(
            "\nEnter your question (or type 'exit' to quit): "
        )

        if user_question.lower() == "exit":
            print("Chatbot closed!")
            break

        question_embedding = model.encode(
            [user_question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=1,
            include=["documents", "distances"]
        )

        distance = results["distances"][0][0]

        if distance > 1.2:
            print("I don't know based on this PDF.")

        else:
            chunk = results["documents"][0][0]

            
            prompt = f"""
            Answer the question using only the text provided below.
            If the answer is not available in the text,
            say: I don't know based on this PDF.

            Text:
            {chunk}

            Question:
            {user_question}
            """

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            print("\nAnswer:")
            print(response.text)