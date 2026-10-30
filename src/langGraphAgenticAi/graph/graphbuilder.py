from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from src.langGraphAgenticAi.state.state import State

from src.langGraphAgenticAi.nodes.basic_chatbot_node import (
    BasicChatbotNode
)

from src.langGraphAgenticAi.tools.search_tool import (
    get_tools,
    create_tool_node
)

from src.langGraphAgenticAi.nodes.chatbot_with_Tool_node import (
    ChatbotWithToolNode
)

from src.langGraphAgenticAi.nodes.ai_news_node import (
    AINewsNode
)


class GraphBuilder:

    def __init__(self, model, tavily_api_key=None):

        if model is None:
            raise ValueError(
                "LLM model is not initialized."
            )

        self.llm = model
        self.tavily_api_key = tavily_api_key

        self.graph_builder = StateGraph(State)

    # ------------------------------------------
    # BASIC CHATBOT
    # ------------------------------------------

    def basic_chatbot_build_graph(self):

        basic_chatbot_node = BasicChatbotNode(self.llm)

        self.graph_builder.add_node(
            "chatbot",
            basic_chatbot_node.process
        )

        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    # ------------------------------------------
    # CHATBOT WITH WEB SEARCH
    # ------------------------------------------

    def chatbot_with_tools_build_graph(self):

        tools = get_tools(
            api_key=self.tavily_api_key
        )

        tool_node = create_tool_node(tools)

        chatbot_with_tool = ChatbotWithToolNode(self.llm)

        chatbot_node = chatbot_with_tool.create_chatbot(tools)

        self.graph_builder.add_node(
            "chatbot",
            chatbot_node
        )

        self.graph_builder.add_node(
            "tools",
            tool_node
        )

        self.graph_builder.add_edge(
            START,
            "chatbot"
        )

        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition
        )

        self.graph_builder.add_edge(
            "tools",
            "chatbot"
        )

    # ------------------------------------------
    # AI NEWS
    # ------------------------------------------

    def ai_news_builder_graph(self):

        ai_news_node = AINewsNode(
            self.llm,
            tavily_api_key=self.tavily_api_key
        )

        self.graph_builder.add_node(
            "fetch_news",
            ai_news_node.fetch_news
        )

        self.graph_builder.add_node(
            "summarize_news",
            ai_news_node.summarize_news
        )

        self.graph_builder.add_node(
            "save_result",
            ai_news_node.save_result
        )

        self.graph_builder.add_edge(
            START,
            "fetch_news"
        )

        self.graph_builder.add_edge(
            "fetch_news",
            "summarize_news"
        )

        self.graph_builder.add_edge(
            "summarize_news",
            "save_result"
        )

        self.graph_builder.add_edge(
            "save_result",
            END
        )

    # ------------------------------------------
    # SETUP GRAPH
    # ------------------------------------------

    def setup_graph(self, usecase: str):

        if not usecase:
            raise ValueError("No use case selected.")

        normalized_usecase = " ".join(
            usecase.strip().lower().split()
        )

        self.graph_builder = StateGraph(State)

        if normalized_usecase == "basic chatbot":

            self.basic_chatbot_build_graph()

        elif normalized_usecase == "chatbot with web search":

            self.chatbot_with_tools_build_graph()

        elif normalized_usecase == "ai news":

            self.ai_news_builder_graph()

        else:

            raise ValueError(
                f"Unsupported use case: {usecase}"
            )

        return self.graph_builder.compile()
