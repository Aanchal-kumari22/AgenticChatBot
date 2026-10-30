from src.langGraphAgenticAi.state.state import State


class BasicChatbotNode:
    """Processes user messages and generates an AI response."""

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        messages = state.get("messages", [])

        if not messages:
            raise ValueError("No messages found in the input state.")

        response = self.llm.invoke(messages)

        return {"messages": [response]}

