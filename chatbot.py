from google import genai
import os

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def start_chat():

    history = []

    system_instruction = """
    You are a friendly AI assistant.
    Answer naturally and help the user.
    """

    while True:

        user_message = input("You: ")

        if user_message.lower() == "exit":
            print("Chat ended. Goodbye!")
            break

        history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=history,
            config={
                "system_instruction": system_instruction
            }
        )

        bot_reply = response.text

        print("Gemini:", bot_reply)

        history.append({
            "role": "model",
            "parts": [{"text": bot_reply}]
        })
