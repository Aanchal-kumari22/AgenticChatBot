import os
import streamlit as st

from src.langGraphAgenticAi.ui.streamlit.loadui import LoadStreamlitUI
from src.langGraphAgenticAi.LLMS.GroqLLM import GroqLLM
from src.langGraphAgenticAi.graph.graphbuilder import GraphBuilder
from src.langGraphAgenticAi.ui.streamlit.displayresult import (
    DisplayResultStreamlit
)


def load_langgraph_agenticai_app():

    # ------------------------------------------
    # LOAD UI
    # ------------------------------------------

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load UI configuration.")
        return

    selected_usecase = user_input.get("selected_usecase")

    if not selected_usecase:
        st.warning("Please select a use case.")
        return

    normalized_usecase = " ".join(
        selected_usecase.strip().lower().split()
    )

    # ------------------------------------------
    # LOAD API KEYS FROM ENVIRONMENT
    # ------------------------------------------

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not groq_api_key:
        st.error(
            "Groq API key is not configured on the server. "
            "Please contact the administrator."
        )
        return

    # Search-based use cases
    search_usecases = {
        "chatbot with web search",
        "ai news"
    }

    if normalized_usecase in search_usecases and not tavily_api_key:
        st.error(
            "Tavily API key is not configured on the server. "
            "Please contact the administrator."
        )
        return

    # Pass environment keys to the existing LLM configuration.
    user_input["GROQ_API_KEY"] = groq_api_key
    user_input["TAVILY_API_KEY"] = tavily_api_key

    # ------------------------------------------
    # VALIDATE LLM PROVIDER
    # ------------------------------------------

    selected_llm = user_input.get("selected_llm")

    if selected_llm != "Groq":
        st.error("Currently only Groq is supported.")
        return

    # ------------------------------------------
    # GET USER MESSAGE
    # ------------------------------------------

    if normalized_usecase == "ai news":

        if not st.session_state.get(
            "IsFetchButtonClicked", False
        ):
            return

        user_message = st.session_state.get(
            "timeframe", "Daily"
        )

    else:

        user_message = st.chat_input(
            "Enter your message:"
        )

        if not user_message:
            return

    # ------------------------------------------
    # INITIALIZE LLM
    # ------------------------------------------

    try:

        llm_config = GroqLLM(
            user_controls_input=user_input
        )

        model = llm_config.get_llm_model()

        if model is None:
            st.error("Groq model initialization failed.")
            return

    except Exception as e:

        st.error(
            f"LLM initialization failed: "
            f"{type(e).__name__}: {e}"
        )
        return

    # ------------------------------------------
    # BUILD GRAPH
    # ------------------------------------------

    try:

        graph_builder = GraphBuilder(
            model=model,
            tavily_api_key=tavily_api_key
        )

        graph = graph_builder.setup_graph(
            selected_usecase
        )

        if graph is None:
            st.error("Graph initialization failed.")
            return

    except Exception as e:

        st.error(
            f"Graph setup failed: "
            f"{type(e).__name__}: {e}"
        )
        return

    # ------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------

    try:

        display = DisplayResultStreamlit(
            selected_usecase,
            graph,
            user_message
        )

        display.display_result_on_ui()

    except Exception as e:

        st.error(
            f"Execution failed: "
            f"{type(e).__name__}: {e}"
        )

    finally:

        if normalized_usecase == "ai news":
            st.session_state.IsFetchButtonClicked = False


if __name__ == "__main__":
    load_langgraph_agenticai_app()
