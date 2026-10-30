from src.langGraphAgenticAi.state.state import State


class ChatbotWithToolNode:

    def __init__(self, model):
        self.llm = model

    def create_chatbot(self, tools):
        """Create a chatbot node that can call tools."""

        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        return chatbot_node
