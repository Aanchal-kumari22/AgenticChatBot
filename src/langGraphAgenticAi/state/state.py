from typing import TypedDict, Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):

    # Chatbot conversation messages
    messages: Annotated[
        List[BaseMessage],
        add_messages
    ]

    # AI News workflow fields
    frequency: str
    news_data: list
    summary: str
    output_file: str
