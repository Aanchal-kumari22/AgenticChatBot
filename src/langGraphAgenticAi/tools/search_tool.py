import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode


def get_tools(api_key=None):
    """
    Initialize and return Tavily search tools
    using the server-side API key.
    """

    # Get API key from the server environment.
    api_key = (
        api_key or os.getenv("TAVILY_API_KEY", "")
    ).strip()

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not configured on the server."
        )

    # Initialize Tavily search tool.
    tools = [
        TavilySearchResults(
            max_results=3,
            search_depth="basic",
            include_answer=True,
            tavily_api_key=api_key
        )
    ]

    return tools


def create_tool_node(tools):
    """
    Create and return a LangGraph ToolNode.
    """

    if not tools:
        raise ValueError(
            "No tools provided to ToolNode."
        )

    return ToolNode(tools=tools)
