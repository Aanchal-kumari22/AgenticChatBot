from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    frequency: str
    news_data: list
    summary: str
    output_file: str
