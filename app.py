from pypdf import PdfReader
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import os
import json
from chatbot import start_chat
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
        print("\nChoose an option:")
        print("1. Ask a question (Chatbot)")
        print("2. Generate a quiz")
        print("3. lets chat")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3): ")

        
        if choice == "1":
            user_question = input(
                "\nEnter your question: "
            )

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
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                print("\nAnswer:")
                print(response.text)

        elif choice == "2":

            prompt = f"""
            Generate exactly 5 multiple-choice questions
            from the following PDF text.

            Each question must have exactly 4 options:
            A, B, C, D.

            Return the output in valid JSON format:

            {{
                "questions": [
                    {{
                        "question": "Question here",
                        "options": {{
                            "A": "Option A",
                            "B": "Option B",
                            "C": "Option C",
                            "D": "Option D"
                        }},
                        "answer": "B"
                    }}
                ]
            }}

            Rules:
            1. Use only the provided PDF text.
            2. Each question must have one correct answer.
            3. Keep the correct answer hidden from the user.
            4. Return only valid JSON.
            5. Generate exactly 5 questions.

            PDF Text:
            {text}
            """

            print("\nGenerating Quiz...")

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            try:
                quiz = json.loads(response.text.strip())

                score = 0

                for i, q in enumerate(
                    quiz["questions"], start=1
                ):

                    print(f"\nQ{i}. {q['question']}")

                    for option, value in q["options"].items():
                        print(f"{option}. {value}")

                    user_answer = input(
                        "\nChoose your answer (A/B/C/D): "
                    ).strip().upper()

                    while user_answer not in ["A", "B", "C", "D"]:
                        print(
                            "Invalid input! Choose A, B, C or D."
                        )

                        user_answer = input(
                            "Your answer: "
                        ).strip().upper()

                    if user_answer == q["answer"].strip().upper():
                        print("Correct!")
                        score += 1

                    else:
                        print(
                            f"Wrong! Correct answer is "
                            f"{q['answer']}."
                        )

                print("\n-------------------------")
                print(
                    f"You got {score} out of "
                    f"{len(quiz['questions'])}."
                )
                print("-------------------------")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print("Quiz generation format error:", e)

        elif choice == "3":
            start_chat()

        elif choice == "4":
            print("Program closed!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
