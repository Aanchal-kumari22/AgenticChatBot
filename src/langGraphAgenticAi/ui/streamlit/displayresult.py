import streamlit as st
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = str(self.usecase).strip().lower()

        # --------------------------------------------------
        # 1. BASIC CHATBOT
        # --------------------------------------------------
        if usecase == "basic chatbot":
            with st.chat_message("user"):
                st.write(self.user_message)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                try:
                    input_state = {
                        "messages": [
                            HumanMessage(content=self.user_message)
                        ]
                    }

                    # Invoke the LangGraph workflow
                    result = self.graph.invoke(input_state)

                    messages = result.get("messages", [])
                    final_answer = None

                    if not isinstance(messages, list):
                        messages = [messages]

                    # Find the last non-empty AI response
                    for message in reversed(messages):
                        if (
                            isinstance(message, AIMessage)
                            and message.content
                            and not getattr(message, "tool_calls", None)
                        ):
                            final_answer = message.content
                            break

                    if final_answer is None:
                        response_placeholder.warning(
                            "The chatbot did not return a final answer. "
                            "Check the terminal for errors."
                        )

                    elif isinstance(final_answer, str):
                        response_placeholder.markdown(final_answer)

                    else:
                        response_placeholder.write(final_answer)

                except Exception as e:
                    response_placeholder.error(
                        f"Failed to generate response: "
                        f"{type(e).__name__}: {e}"
                    )

        # --------------------------------------------------
        # 2. CHATBOT WITH WEB SEARCH
        # --------------------------------------------------
        elif usecase == "chatbot with web search":
            try:
                initial_state = {
                    "messages": [
                        HumanMessage(content=self.user_message)
                    ]
                }

                # Invoke the graph and retrieve its messages
                result = self.graph.invoke(initial_state)
                messages = result.get("messages", [])

                if not isinstance(messages, list):
                    messages = [messages]

                final_answer = None

                # Display the conversation in order
                for message in messages:
                    if isinstance(message, HumanMessage):
                        with st.chat_message("user"):
                            st.write(message.content)

                    elif isinstance(message, ToolMessage):
                        with st.chat_message("assistant"):
                            st.caption("Web search tool result")
                            st.write(message.content)

                    elif isinstance(message, AIMessage):
                        # Ignore intermediate tool-calling messages
                        if getattr(message, "tool_calls", None):
                            continue

                        if message.content:
                            with st.chat_message("assistant"):
                                if isinstance(message.content, str):
                                    st.markdown(message.content)
                                else:
                                    st.write(message.content)

                            final_answer = message.content

                # Handle an empty response
                if final_answer is None:
                    st.warning(
                        "The chatbot did not return a final answer. "
                        "Check the terminal for errors."
                    )

            except Exception as e:
                st.error(
                    f"Failed to generate response: "
                    f"{type(e).__name__}: {e}"
                )

        # --------------------------------------------------
        # 3. AI NEWS
        # --------------------------------------------------
        elif usecase == "ai news":
            try:
                frequency = str(self.user_message).strip()

                if not frequency:
                    st.warning("Please provide a news frequency.")
                    return

                with st.spinner("Fetching and summarizing news..."):
                    # Keep the input format shown in your original code
                    result = self.graph.invoke(
                        {"messages": frequency}
                    )

                # Find the generated Markdown summary
                news_path = (
                    Path.cwd()
                    / "AINews"
                    / f"{frequency.lower()}_summary.md"
                )

                if not news_path.is_file():
                    st.error(
                        "News summary was not generated or the file "
                        f"was not found: {news_path}"
                    )
                    return

                markdown_content = news_path.read_text(
                    encoding="utf-8"
                ).strip()

                if markdown_content:
                    st.markdown(
                        markdown_content,
                        unsafe_allow_html=False
                    )
                else:
                    st.warning("The generated news summary is empty.")

            except Exception as e:
                st.error(
                    f"An error occurred: {type(e).__name__}: {e}"
                )

        # --------------------------------------------------
        # 4. UNSUPPORTED USE CASE
        # --------------------------------------------------
        else:
            st.error(f"Unsupported use case: {self.usecase}")
