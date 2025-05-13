# memory.py

from langchain.memory import ConversationBufferMemory

class ChatMemory:
    def __init__(self):
        # Initialize LangChain memory to store conversation context
        self.memory = ConversationBufferMemory(memory_key="history")

    def add_turn(self, user_input, bot_response):
        # Add user input and bot response to LangChain memory
        self.memory.save_context({"input": user_input}, {"output": bot_response})

    def get_prompt(self, new_input):
        # Retrieve the conversation history so far
        return self.memory.load_memory_variables({})["history"] + f"User: {new_input}\nBot:"
