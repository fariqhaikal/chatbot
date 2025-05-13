# main.py

from model_loader import load_model
from memory import ChatMemory
import wikipedia
from langchain.agents import Tool
from langchain.agents import initialize_agent
from chat import generate_response  # Import the function directly from chat.py

# Wikipedia Tool
def wikipedia_query(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=1)
        return summary
    except Exception as e:
        return f"Sorry, I couldn't find information on {query}. Error: {e}"

# Define the Wikipedia Tool for LangChain agent
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_query,
    description="Fetches a summary from Wikipedia based on a query."
)

# Main function to run the chatbot
def main():
    tokenizer, model = load_model()  # Load the local model
    memory = ChatMemory()  # Use LangChain memory for conversation history

    print("🤖 Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user input to the memory (you can store and retrieve past interactions)
        memory.add_turn(user_input, "")  # No bot response yet

        # Retrieve conversation history
        prompt = memory.get_prompt(user_input)

        # Generate the response from the model by calling the function in chat.py
        bot_response = generate_response(prompt, tokenizer, model)

        # Store the bot's response in memory
        memory.add_turn(user_input, bot_response)

        # Fetch the Wikipedia summary (if available) for additional information
        wikipedia_response = wikipedia_query(user_input)

        # Output the responses
        print("Bot (Wikipedia):", wikipedia_response)
        print("Bot (GPT-2):", bot_response)

if __name__ == "__main__":
    main()
