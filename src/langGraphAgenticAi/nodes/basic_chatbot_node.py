from src.langGraphAgenticAi.state.state import State


class BasicChatbotNode:
    """
    Processes user messages and generates an AI response.
    """

    def __init__(self, model):

        if model is None:
            raise ValueError("LLM model is not initialized.")

        self.llm = model

    def process(self, state: State) -> dict:

        messages = state.get("messages", [])

        if not messages:
            raise ValueError(
                "No messages found in the input state."
            )

        try:

            response = self.llm.invoke(messages)

            if response is None:
                raise ValueError(
                    "The LLM returned an empty response."
                )

            return {
                "messages": [response]
            }

        except Exception as e:

            raise RuntimeError(
                f"Basic chatbot execution failed: "
                f"{type(e).__name__}: {e}"
            ) from e
