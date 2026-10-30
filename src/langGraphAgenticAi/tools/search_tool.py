import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode


def get_tools(api_key=None):

    """
    Initialize and return Tavily search tools.
    """

    api_key = api_key or os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "Tavily API key is missing. "
            "Please enter your Tavily API key."
        )

    os.environ["TAVILY_API_KEY"] = api_key

    tools = [
        TavilySearchResults(
            max_results=3,
            search_depth="basic",
            include_answer=True
        )
    ]

    return tools


def create_tool_node(tools):

    """
    Create and return a LangGraph ToolNode.
    """

    if not tools:
        raise ValueError("No tools provided to ToolNode.")

    return ToolNode(tools=tools)
