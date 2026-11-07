from src.langGraphAgenticAi.state.state import State


class ChatbotWithToolNode:

    def __init__(self, model):

        if model is None:
            raise ValueError(
                "LLM model is not initialized."
            )

        self.llm = model

    def create_chatbot(self, tools):
        """
        Create a chatbot node that can invoke
        Tavily web search tools.
        """

        if not tools:
            raise ValueError(
                "No tools provided to the chatbot."
            )

        # Bind Tavily tools to the Groq LLM.
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):

            messages = state.get("messages", [])

            if not messages:
                raise ValueError(
                    "No messages found in the input state."
                )

            try:

                response = llm_with_tools.invoke(messages)

                if response is None:
                    raise ValueError(
                        "The LLM returned no response."
                    )

                return {
                    "messages": [response]
                }

            except Exception as e:

                raise RuntimeError(
                    f"Web search chatbot failed: "
                    f"{type(e).__name__}: {e}"
                ) from e

        return chatbot_node
