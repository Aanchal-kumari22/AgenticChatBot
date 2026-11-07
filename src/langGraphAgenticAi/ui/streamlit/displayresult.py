import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)


class DisplayResultStreamlit:

    def __init__(self, usecase, graph, user_message):

        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):

        usecase = " ".join(
            str(self.usecase).strip().lower().split()
        )

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
                            HumanMessage(
                                content=self.user_message
                            )
                        ]
                    }

                    result = self.graph.invoke(input_state)

                    messages = result.get("messages", [])

                    if not isinstance(messages, list):
                        messages = [messages]

                    final_answer = None

                    for message in reversed(messages):

                        if (
                            isinstance(message, AIMessage)
                            and message.content
                            and not getattr(
                                message, "tool_calls", None
                            )
                        ):
                            final_answer = message.content
                            break

                    if final_answer is None:

                        response_placeholder.warning(
                            "The chatbot did not return a final answer."
                        )

                    elif isinstance(final_answer, str):

                        response_placeholder.markdown(
                            final_answer
                        )

                    else:

                        response_placeholder.write(
                            final_answer
                        )

                except Exception as e:

                    response_placeholder.error(
                        f"Failed to generate response: "
                        f"{type(e).__name__}: {e}"
                    )

        # --------------------------------------------------
        # 2. CHATBOT WITH WEB SEARCH
        # --------------------------------------------------

        elif usecase == "chatbot with web search":

            with st.chat_message("user"):
                st.write(self.user_message)

            try:

                initial_state = {
                    "messages": [
                        HumanMessage(
                            content=self.user_message
                        )
                    ]
                }

                with st.spinner(
                    "Searching the web and generating an answer..."
                ):

                    result = self.graph.invoke(initial_state)

                messages = result.get("messages", [])

                if not isinstance(messages, list):
                    messages = [messages]

                final_answer = None

                # Find the final AI response.
                for message in messages:

                    if isinstance(message, AIMessage):

                        if (
                            message.content
                            and not getattr(
                                message, "tool_calls", None
                            )
                        ):
                            final_answer = message.content

                if final_answer is not None:

                    with st.chat_message("assistant"):

                        if isinstance(final_answer, str):
                            st.markdown(final_answer)
                        else:
                            st.write(final_answer)

                    # Display search results separately.
                    tool_messages = [
                        message
                        for message in messages
                        if isinstance(message, ToolMessage)
                    ]

                    if tool_messages:

                        with st.expander(
                            "View web search results"
                        ):

                            for tool_message in tool_messages:
                                st.write(tool_message.content)

                else:

                    st.warning(
                        "The chatbot did not return a final answer."
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

                frequency = str(
                    self.user_message
                ).strip().lower()

                if frequency not in {
                    "daily",
                    "weekly",
                    "monthly"
                }:

                    st.warning(
                        "Please select Daily, Weekly or Monthly."
                    )
                    return

                input_state = {
                    "frequency": frequency,
                    "news_data": [],
                    "summary": ""
                }

                with st.spinner(
                    f"Fetching {frequency} AI news and "
                    "generating summaries..."
                ):

                    result = self.graph.invoke(input_state)

                # Get summary directly from graph output.
                summary = result.get("summary", "")

                if summary:

                    st.success(
                        f"{frequency.capitalize()} AI News "
                        "generated successfully!"
                    )

                    st.markdown("---")

                    st.markdown(
                        summary,
                        unsafe_allow_html=False
                    )

                    # Show saved file location.
                    output_file = result.get(
                        "output_file"
                    )

                    if output_file:

                        st.caption(
                            f"Summary saved at: {output_file}"
                        )

                else:

                    st.warning(
                        "No AI news summary was generated."
                    )

            except Exception as e:

                st.error(
                    f"AI News generation failed: "
                    f"{type(e).__name__}: {e}"
                )

        # --------------------------------------------------
        # 4. UNSUPPORTED USE CASE
        # --------------------------------------------------

        else:

            st.error(
                f"Unsupported use case: {self.usecase}"
            )
